#coding=utf-8
import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Tanh, Dropout, MaxPool2d, \
    AdaptiveAvgPool2d, Sequential, Module
from collections import namedtuple, OrderedDict


# Support: ['IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']

tanh_act = False

class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


# def l2_norm(input, axis=1):
#     norm = torch.norm(input, 2, axis, True)
#     output = torch.div(input, norm)
#
#     return output


class SEModule(Module):
    def __init__(self, channels, reduction, tanh_act=tanh_act):
        super(SEModule, self).__init__()
        self.tanh_act = tanh_act
        self.avg_pool = AdaptiveAvgPool2d(1)    # https://blog.csdn.net/u013382233/article/details/85948695
        self.fc1 = Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        nn.init.xavier_uniform_(self.fc1.weight.data)

        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = Sigmoid()
        self.tanh = Tanh()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.tanh_act:
            x = self.tanh(x) + 1    # 参airFace中的设置，2019.8.19 add
        else:
            x = self.sigmoid(x)
        return module_input * x


class basic_block_IR(Module):
    expansion = 1
    def __init__(self, in_channel, depth, stride, match, SE):
        super(basic_block_IR, self).__init__()
        self.shortcut_layer = None
        if not match:
            self.shortcut_layer = Sequential(Conv2d(in_channel, depth, (1, 1), stride, bias=False), BatchNorm2d(depth))
        self.res_layer = Sequential(
                                    BatchNorm2d(in_channel),
                                    Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
                                    BatchNorm2d(depth),            # xk: add BN according to insightface paper
                                    PReLU(depth),
                                    Conv2d(depth, depth, (3, 3), stride, 1, bias=False),   # 第2个conv的s可为2，ResNet中是第一个
                                    BatchNorm2d(depth)   )         # 最好按照ResNet方式，init中定义各模块的对象，forward中再依次串起来，若按照这里用Sequential直接串起来，
                                                                   # 则modules.named_parameters()返回的pname中看不出来是bn还是conv，会被自动用0,1...数字索引替代
        if SE:
            self.res_layer.add_module(str(len(self.res_layer)), SEModule(depth, 16))   # 参Sequential定义编写

    def forward(self, x):
        shortcut = x
        if self.shortcut_layer is not None:
            shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class bottleneck_block_IR(Module):
    expansion = 4
    def __init__(self, in_channel, depth, stride, match, SE):
        super(bottleneck_block_IR, self).__init__()
        self.shortcut_layer = None
        if not match:
            self.shortcut_layer = Sequential(Conv2d(in_channel, depth*self.expansion, (1, 1), stride, bias=False), BatchNorm2d(depth))
        self.res_layer = Sequential(
                                    BatchNorm2d(in_channel),
                                    Conv2d(in_channel, depth, (1, 1), 1, bias=False),
                                    BatchNorm2d(depth),    # xk: add BN according to insightface paper
                                    PReLU(depth),
                                    Conv2d(depth, depth, (3, 3), 1, 1, bias=False),
                                    BatchNorm2d(depth),
                                    PReLU(depth),
                                    Conv2d(depth, depth * self.expansion, (1, 1), stride, bias=False),  # 第3个conv的s可为2，ResNet中是第一个
                                    BatchNorm2d(depth * self.expansion),
                                    )
        if SE:
            self.res_layer.add_module(str(len(self.res_layer)), SEModule(depth, 16))   # 参Sequential定义编写

    def forward(self, x):
        shortcut = x
        if self.shortcut_layer is not None:
            shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class Block(namedtuple('Block', ['in_channel', 'depth', 'stride', 'match', 'SE'])):
    '''A named tuple describing a ResNet block.'''

# note: conv3_1,conv4_1,conv5_1的match均为False（输入输出的特征图大小不同，且通道数不同）
#       conv2_1的match不定（输入输出的特征图大小可能不同，视conv1而定，这里是match=False）,具体分析如下：
# insightface中，为了延缓降低分辨率，选择conv1=3x3且s=1, conv2_1的输入112x112，输出56x56，通道均为64，故match=False, conv2_1须设置s=2
# 若将Backbone()的input_layer改用s=2，则conv2_1的match=True（输入输出通道均为64，特征图均为56x56）===> 此时须注意改conv2_x的s=1
# 另：ResNet中，conv2_1不进行降采样(s=1)，其输入输出为56x56，通道数均为64，match=True（对应论文中shortcut path为实线）
def get_basic_blocks(in_channel, depth, num_units, stride=2, match=False, SE=False):

    return [Block(in_channel, depth, stride, match, SE)] + [Block(depth, depth, 1, True, SE) for i in range(num_units - 1)]


def get_bottleneck_blocks(in_channel, depth, num_units, stride=2, match=False, SE=False):
    expansion = 4
    return [Block(in_channel, depth, stride, match, SE)] + [Block(depth*expansion, depth, 1, True, SE) for i in range(num_units - 1)]


def get_layers(num_layers, SE):
    if num_layers == 50:
        layers = [
            get_basic_blocks(in_channel=64, depth=64, num_units=3, SE=SE),
            get_basic_blocks(in_channel=64, depth=128, num_units=4, SE=SE),
            get_basic_blocks(in_channel=128, depth=256, num_units=14, SE=SE),
            get_basic_blocks(in_channel=256, depth=512, num_units=3, SE=SE)
        ]
    elif num_layers == 100:
        layers = [
            get_basic_blocks(in_channel=64, depth=64, num_units=3, SE=SE),
            get_basic_blocks(in_channel=64, depth=128, num_units=13, SE=SE),
            get_basic_blocks(in_channel=128, depth=256, num_units=30, SE=SE),
            get_basic_blocks(in_channel=256, depth=512, num_units=3, SE=SE)
        ]
    elif num_layers == 152:
        layers = [
            get_bottleneck_blocks(in_channel=64, depth=64, num_units=3, SE=SE),
            get_bottleneck_blocks(in_channel=256, depth=128, num_units=8, SE=SE),
            get_bottleneck_blocks(in_channel=512, depth=256, num_units=36, SE=SE),
            get_bottleneck_blocks(in_channel=1024, depth=512, num_units=3, SE=SE)
        ]

    return layers


class Backbone(Module):
    def __init__(self, input_size, num_layers, mode='ir',opt='E'):
        super(Backbone, self).__init__()
        assert input_size[0] in [112, 224], "input_size should be [112, 112] or [224, 224]"
        assert num_layers in [50, 100, 152], "num_layers should be 50, 100 or 152"
        assert mode in ['ir', 'ir_se'], "mode should be ir or ir_se"
        self.opt = opt
        se = False
        if mode == 'ir_se':
            se = True
        layers = get_layers(num_layers, se)
        unit_module = basic_block_IR
        if num_layers == 152:
            unit_module = bottleneck_block_IR

        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))                    # 相比原resnet，这里没有max pool降采样
        if input_size[0] == 112:
            if self.opt == 'E':
                self.output_layer = Sequential(OrderedDict([        # 改用OrderedDict，方便分组wd时引用out_fc
                                            ('out_bn1', BatchNorm2d(512 * unit_module.expansion)),
                                            ('out_dropout', Dropout()),
                                            ('out_flat', Flatten()),
                                            ('out_fc', Linear(512 * unit_module.expansion * 7 * 7, 512)),
                                            ('out_bn2', BatchNorm1d(512))])
                                         )
            elif self.opt == 'pseudo_fc':
                self.output_layer = Sequential(OrderedDict([
                                             ('out_avgpool', nn.AdaptiveAvgPool2d((1, 1))),
                                             ('out_conv2', nn.Conv2d(512 * unit_module.expansion, 16 * unit_module.expansion, kernel_size=1, bias=False)),
                                             ('out_bn2', nn.BatchNorm1d(1296 * unit_module.expansion)),
                                             ('out_fc', nn.Linear(1296 * unit_module.expansion, 512))
                                        ]))
        else:       # 224x224
            self.output_layer = Sequential(OrderedDict([
                                            ('out_bn1', BatchNorm2d(512 * unit_module.expansion)),
                                            ('out_dropout', Dropout()),
                                            ('out_flat', Flatten()),
                                            ('out_fc', Linear(512 * unit_module.expansion * 14 * 14, 512)),
                                            ('out_bn2', BatchNorm1d(512))])
                                          )
        modules = []
        for layer in layers:
            for block in layer:
                modules.append( unit_module(block.in_channel,
                                            block.depth,
                                            block.stride,
                                            block.match,
                                            block.SE )
                                )
        self.body = Sequential(*modules)

        self._initialize_weights()

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        if self.opt == 'E':
            x = self.output_layer(x)

        # 2019.9.10 add
        # ref: https://github.com/SomeoneDistant/Lightweight-Face-Recognition/blob/master/ResNet/model.py
        elif self.opt == 'pseudo_fc':
            x1 = self.output_layer.out_avgpool(x)
            x2 = self.output_layer.out_conv2(x)
            x2 = x2.reshape(x2.size(0), -1, 1, 1)
            x = torch.cat((x1, x2), 1)
            x = x.squeeze()
            x = self.output_layer.out_bn2(x)
            x = self.output_layer.out_fc(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


def IR_50(input_size,opt):
    """Constructs a ir-50 model.
    """
    model = Backbone(input_size, 50, 'ir', opt=opt)

    return model


def IR_101(input_size,opt):
    """Constructs a ir-101 model.
    """
    model = Backbone(input_size, 100, 'ir', opt=opt)

    return model


def IR_152(input_size,opt):
    """Constructs a ir-152 model.
    """
    model = Backbone(input_size, 152, 'ir', opt=opt)

    return model


def IR_SE_50(input_size,opt):
    """Constructs a ir_se-50 model.
    """
    model = Backbone(input_size, 50, 'ir_se', opt=opt)

    return model


def IR_SE_101(input_size,opt):
    """Constructs a ir_se-101 model.
    """
    model = Backbone(input_size, 100, 'ir_se', opt=opt)

    return model


def IR_SE_152(input_size,opt):
    """Constructs a ir_se-152 model.
    """
    model = Backbone(input_size, 152, 'ir_se', opt=opt)

    return model
