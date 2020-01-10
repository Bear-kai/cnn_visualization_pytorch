# 人脸识别模型的现象：
# 1. 无预处理时，大模型的测试集性能稍降低几个点，小模型直接崩（近似0）
# 2. 有预处理时，在验证集上表现接近，甚至小模型略优时，在测试集上大模型还是比小模型高好几个点
# （note by huhui: 可能是验证集难度不够，不足以在验证集上区分二者性能）
# 初步结论：大模型的鲁棒性优于小模型
# 鉴于以上两种现象，尤其是现象1，尝试从以下4个方面进行探索，以期望更深入地理解模型，继而优化模型。
# 1.统计各层权值范数;
# 2.给定一批样本，统计梯度信息(ref AAAI19-GHM) ---> GHMC暂失败！
# 3.给定一批样本，统计特征范数(ref ICCV19-LargeNorm);   ---> 待测，训练网络时添加特征范数约束
# 4.特征可视化(ref ECCV14 + ArXiv14)  ---> 整理构建系列可视化工具

import torch
import torch.nn as nn
import os
import sys
from backbone.my_model_irse import IR_50
from backbone.model_irse import IR_50_pub
from backbone.MobileFaceNet import MobileFaceNet, MobileFaceNet_air
import matplotlib.pyplot as plt


def gen_backbone(backbone_name, input_size, CBAM=False, tanh_act=False):
    if backbone_name == 'IR_50':
        return IR_50(input_size,opt='E')
    elif backbone_name in ['MobileFaceNet', 'MobileFaceNet2x_1','MobileFaceNet2x_2']:
        return MobileFaceNet(setting_str=backbone_name)
    elif backbone_name == 'MobileFaceNet_air':
        return MobileFaceNet_air(CBAM=CBAM, tanh_act=tanh_act)
    elif backbone_name == 'IR_50_pub':
        return IR_50_pub(input_size)
    else:
        raise ValueError("backbone_name not supported")


def load_pth_model_pub(checkpoint_path, backbone):
    print("=" * 60)
    if os.path.isfile(checkpoint_path):
        print("loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        backbone.load_state_dict(checkpoint)        # ***
    else:
        print("No Checkpoint exists at '{}'".format(checkpoint_path))
        sys.exit(0)
    print("=" * 60)
    return backbone


def load_pth_model(checkpoint_path, backbone):
    print("=" * 60)
    if os.path.isfile(checkpoint_path):
        print("loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        backbone.load_state_dict(checkpoint['backbone'])
        head_norm = torch.norm(checkpoint['head']['weight'].data)
    else:
        print("No Checkpoint exists at '{}'. Train from Scratch".format(checkpoint_path))
    print("=" * 60)
    return backbone, head_norm


def collect_conv_weight_norm(model):
    weight_dic = {'conv_weight_norm':[], 'conv_weight_size':[],
                  'bn_weight_norm':[],'bn_weight_size':[],
                  'fc_weight_norm':[],'fc_weight_size':[]
                  }
    bias_dic = {'conv_bias_norm':[], 'conv_bias_size':[],
                'bn_bias_norm':[], 'bn_bias_size':[],
                'fc_bias_norm':[], 'fc_bias_size':[]
                }
    running_dic = {'mean':[], 'var':[]}
    for m in model.modules():
        # backbone.modules()第一个返回backbone全体，然后是各Sequential, 各模块Conv2d, BatchNorm2d等
        if isinstance(m, nn.Conv2d):
            weight_dic['conv_weight_norm'].append(torch.norm(m.weight.data))
            weight_dic['conv_weight_size'].append(tuple(m.weight.shape))
            if m.bias is not None:
                bias_dic['conv_bias_norm'].append(torch.norm(m.bias.data))
                bias_dic['conv_bias_size'].append(tuple(m.weight.shape))
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            weight_dic['bn_weight_norm'].append(torch.norm(m.weight.data))
            weight_dic['bn_weight_size'].append(tuple(m.weight.shape))
            bias_dic['bn_bias_norm'].append(torch.norm(m.bias.data))
            bias_dic['bn_bias_size'].append(tuple(m.weight.shape))
            running_dic['mean'].append(torch.mean(m._buffers['running_mean']))
            running_dic['var'].append(torch.mean(m._buffers['running_var']))
        elif isinstance(m, nn.Linear):
            weight_dic['fc_weight_norm'].append(torch.norm(m.weight.data))
            weight_dic['fc_weight_size'].append(tuple(m.weight.shape))
            if m.bias is not None:
                bias_dic['fc_bias_norm'].append(torch.norm(m.bias.data))
                bias_dic['fc_bias_size'].append(tuple(m.weight.shape))
    return weight_dic, bias_dic, running_dic


def draw_and_save(weight_dic, bias_dic, running_dic, name):
    # --------------------------
    fig, axs = plt.subplots(2, 4, figsize=(12,6))
    axs[0,0].plot(weight_dic['conv_weight_norm'])
    axs[0,0].grid()
    axs[0,0].set_title('conv_weight_norm')

    axs[0,1].plot(weight_dic['bn_weight_norm'])
    axs[0,1].grid()
    axs[0,1].set_title('bn_weight_norm')

    axs[0,2].plot(weight_dic['fc_weight_norm'])
    axs[0,2].grid()
    axs[0,2].set_title('fc_weight_norm')

    axs[0,3].plot(running_dic['mean'])
    axs[0,3].grid()
    axs[0,3].set_title('running_mean')

    axs[1,0].plot(bias_dic['conv_bias_norm'])
    axs[1,0].grid()
    axs[1,0].set_title('conv_bias_norm')

    axs[1,1].plot(bias_dic['bn_bias_norm'])
    axs[1,1].grid()
    axs[1,1].set_title('bn_bias_norm')  # , fontsize=10

    axs[1,2].plot(bias_dic['fc_bias_norm'])
    axs[1,2].grid()
    axs[1,2].set_title('fc_bias_norm')

    axs[1,3].plot(running_dic['var'])
    axs[1,3].grid()
    axs[1,3].set_title('running_var')

    # fig.suptitle('%s' % name)  # , fontsize=10, y=1.08   加上大标题会有重叠，调整不过来，放弃...
    plt.tight_layout()
    plt.savefig('%s.jpg'%name)
    # # --------------------------
    # weight_keys = ['conv_weight_size', 'bn_weight_size', 'fc_weight_size']
    # bias_keys = ['conv_bias_size', 'bn_bias_size', 'fc_bias_size']
    # with open('%s.txt'%name, 'a') as file_txt:
    #     file_txt.write('\n\n')
    #     for i,key in enumerate(weight_keys):
    #         # file_txt.write('===== %s ====='%(key) + '\n')
    #         # for term in weight_dic[key]:
    #         #     file_txt.write(str(term) + '\n')
    #         # file_txt.write('===== %s =====' % bias_keys[i] + '\n')
    #         # for term in bias_dic[bias_keys[i]]:
    #         #     file_txt.write(str(term) + '\n')
    #         if len(bias_dic[bias_keys[i]]):
    #             file_txt.write('===== %s | %s =====' % (key, bias_keys[i]) + '\n')
    #             for j,term in enumerate(weight_dic[key]):
    #                 file_txt.write(str(term) + '\t' + str(bias_dic[bias_keys[i]][j]) + '\n')
    #         else:
    #             file_txt.write('===== %s | have no %s =====' % (key, bias_keys[i]) + '\n')
    #             for term in weight_dic[key]:
    #                 file_txt.write(str(term) + '\n')

    return 0


if __name__ == '__main__':
    # ------------------------- 1
    backbone_name = 'IR_50_pub'  # 'MobileFaceNet_air' # 'IR_50' #      ******
    head_name = 'private_asia' #'ms1m_epoch120' # 'AirFace' # 'CosFace' #       ******
    checkpoint_path = './buffer_model/MobileFaceNet_air_CBAM_AirFace_2019-10-14_02_56_46-done/checkpoint.tar'   # ******

    input_size = [112, 112]
    CBAM = False
    tanh_act = False
    # ------------------------- 2
    backbone = gen_backbone(backbone_name, input_size, CBAM, tanh_act)

    if backbone_name in ['IR_50_pub']:
        # model_path = './model_zoo/ir50_ms1m_epoch120/backbone_ir50_ms1m_epoch120.pth'     # *****
        model_path = './model_zoo/ir50_private_asia/backbone_ir50_private_asia.pth'  # *****
        backbone = load_pth_model_pub(model_path, backbone)
        head_norm = torch.tensor(0.0)
    else:
        backbone, head_norm = load_pth_model(checkpoint_path, backbone)

    # ------------------------- 3
    weight_dic, bias_dic, running_dic = collect_conv_weight_norm(backbone)
    draw_and_save(weight_dic, bias_dic, running_dic, backbone_name+'_'+head_name+'_HeadNorm-%.3f'%head_norm.item())
    print('done')
