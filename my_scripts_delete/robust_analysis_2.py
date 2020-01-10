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

import mxnet as mx
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np
import math
import random as rd
import os
from backbone.my_model_irse import IR_50
from backbone.model_irse import IR_50_pub
from backbone.MobileFaceNet import MobileFaceNet, MobileFaceNet_air
from util.image_iter_rec import FaceImageIter
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import datetime
from util.utils import ccrop_batch, l2_norm


class AirFace(nn.Module):

    def __init__(self, in_features, out_features, device_id, s=64.0, m=0.40):
        super(AirFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device_id = device_id
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        if self.device_id == None:  # cpu上操作
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))  # NxC
        else:  # gpu上操作
            cosine = F.linear(F.normalize(input.cuda(self.device_id[0])),
                              F.normalize(self.weight.cuda(self.device_id[0])))
        # --------------------------- linear target logit  ------------------------------
        cosine = torch.clamp(cosine, -1.0, 1.0)
        theta = torch.acos(cosine)  # theta  is between [0,pi]
        theta_m = 1.0 - 2 * (theta + self.m) / math.pi  # theta_m is between [-1,1]     可以考虑继续添加约束，当theta+m>pi时
        theta = 1.0 - 2 * theta / math.pi
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size())
        if self.device_id != None:
            one_hot = one_hot.cuda(self.device_id[0])
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * theta_m) + ((1.0 - one_hot) * theta)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        return output, cosine


class GaussianSim(object):
    def __init__(self, a=1.2, mu=0, sigma=1):
        self.a = a
        self.mu = mu
        self.sigma = 2*sigma

    def __call__(self, x, y):
        out = self.a * torch.exp(-(x-y-self.mu)**2/self.sigma)
        return out


class ArcNegFace(nn.Module):

    def __init__(self, in_features, out_features, device_id, s=64.0, m=0.30, a=1.2, mu=0, sigma=1, easy_margin=False):
        super(ArcNegFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features    # num_class
        self.device_id = device_id
        self.s = s
        self.m = m
        self.cal_sim = GaussianSim(a, mu, sigma)

        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)  # cos(pi-m) = -cos(m)
        self.mm = math.sin(math.pi - m) * m  # sin(pi-m)*m = sim(m)*m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        if self.device_id == None:
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        else:
            cosine = F.linear(F.normalize(input.cuda(self.device_id[0])),
                              F.normalize(self.weight.cuda(self.device_id[0])))

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))  # N*C
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(\theta+m)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)  # 为真时，跟0比：若theta在[0,pi/2)内，则加margin，即取phi，否则不强行施加margin拉近wi和xi，防止xi是噪声（自理解）
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)  # 否则跟self.th比： 若theta<pi-m，则加margin，即取phi
        # --------------------------- convert label to one-hot --------------------------- 否则取cos(theta)-m*sin(m) 费解? 仿cosface的cos(theta)-m ?
        one_hot = torch.zeros(cosine.size())
        if self.device_id != None:
            one_hot = one_hot.cuda(self.device_id[0])
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # ------------- 2019.10.14 ------------ ArcNegFace须修饰非目标类
        cosine_orig = cosine.clone()        # 2019.10.23 add, 对theta的统计是针对原始cosine而非修改后的！
        num = sine.shape[0]
        Cy = torch.max(one_hot * phi, dim=1)[0].reshape(num,1).repeat(1,self.out_features)    # 0:val, 1:index  另expand
        t_jy = self.cal_sim(cosine, Cy)
        cosine = t_jy * cosine + t_jy - 1
        del Cy, t_jy
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        # ------------- 2019.9.30从adapt中copy：统计mini-batch中\theta的均值方差
        cos_theta = torch.masked_select(cosine_orig, one_hot.type(torch.bool))  # ===== torch_1.0, 数据类型是torch.uint8时,tensor用作mask ---> is deprecated, use bool instead in torch_1.2
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)  # tensor.type()返回tensor数据类型，tensor.type(new_type)返回新类型
        theta1 = torch.acos(cos_theta) * (180 / math.pi)  # tensor.type_as(tensor1)将tensor的数据类型转换为跟tensor1一致
        del cos_theta
        return output, cosine_orig   #, theta1


class CosFace(nn.Module):

    def __init__(self, in_features, out_features, device_id, s=64.0, m=0.35):
        super(CosFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device_id = device_id
        self.s = s
        self.m = m

        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        if self.device_id == None:
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        else:
            cosine = F.linear(F.normalize(input.cuda(self.device_id[0])),
                              F.normalize(self.weight.cuda(self.device_id[0])))

        phi = cosine - self.m
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size())
        if self.device_id != None:
            one_hot = one_hot.cuda(self.device_id[0])
        # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)  # 变成one-hot编码，1表示按列填充，中间是索引，最后是填充值
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        return output, cosine


def gen_head(head_name, embedding_size, num_class, GPU_id):
    if head_name == 'CosFace':
        return CosFace(in_features=embedding_size, out_features=num_class, device_id=GPU_id)
    elif head_name == 'AirFace':
        return AirFace(in_features=embedding_size, out_features=num_class, device_id=GPU_id)
    elif head_name == 'ArcNegFace':
        return ArcNegFace(in_features=embedding_size, out_features=num_class, device_id=GPU_id)
    else:
        raise ValueError("head_name not supported")


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


def load_pth_model(checkpoint_path, backbone, head=None):
    print("=" * 60)
    if os.path.isfile(checkpoint_path):
        print("loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        backbone.load_state_dict(checkpoint['backbone'])
        if head:
            head.load_state_dict(checkpoint['head'], strict=False)
    else:
        print("No Checkpoint exists at '{}'. Train from Scratch".format(checkpoint_path))
    print("=" * 60)
    return backbone, head


def get_train_loader_rec(DATA_ROOT,BATCH_SIZE,img_size=(112,112)):                    # 2019.9.2 add
    path_imgrec = os.path.join(DATA_ROOT, "train.rec")
    train_loader = FaceImageIter(
        batch_size=BATCH_SIZE,
        data_shape=(3,img_size[0],img_size[1]),
        path_imgrec=path_imgrec,
        shuffle=True,
        rand_mirror=True,
        resize_rand_crop=True,              # 2019.10.10 add
        mean=(127.5, 127.5, 127.5),          # 为None时不处理；存疑：insightface训练代码为何都设为None，不用预处理？！--> 网络内部有处理
        cutoff=False,
        color_jittering=0,
        images_filter=0,
    )
    # train_loader = mx.io.PrefetchingIter(train_loader)
    return train_loader


from mxnet import recordio
def show_samples_of_one_class(label,path_imgrec,path):
    assert isinstance(label,int)
    path_class = os.path.join(path, str(label))
    if not os.path.exists(path_class):
        os.makedirs(path_class)
    path_imgidx = path_imgrec[0:-4] + ".idx"
    imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')  # pylint: disable=redefined-variable-type
    s = imgrec.read_idx(0)
    header, _ = recordio.unpack(s)
    print('row index range of class', header.label)
    identities = range(int(header.label[0]), int(header.label[1]))    # 类别的行索引
    class_row_ind = identities[label]
    s = imgrec.read_idx(class_row_ind)
    header, _ = recordio.unpack(s)
    samples_inds = range(int(header.label[0]),int(header.label[1]))
    for ii, idx in enumerate(samples_inds):
        s = imgrec.read_idx(idx)
        header, x = recordio.unpack(s)
        x = mx.image.imdecode(x).asnumpy()  # mx.NDArray --> numpy.ndarray
        x = x[:, :, ::-1]  # rgb to bgr
        cv2.imwrite(os.path.join(path_class, str(ii+1) + '.jpg'), x)
    return 0


# def choose_sample(cur_grad,num=1,small=0.1, mid=(0.3,0.7), large=0.95):     # 选中区域中的一个样本
#     ind_dic = {'small': [], 'middle': [], 'large': []}
#     # ----- small grad
#     inds = np.where(cur_grad < small)[0]
#     if len(inds):
#         ind_dic['small'].append(rd.sample(list(inds),num)[0])  # choose one sample to show
#     # ----- large grad
#     inds = np.where(cur_grad > large)[0]
#     if len(inds):
#         ind_dic['large'].append(rd.sample(list(inds),num)[0])
#     # ----- middle grad
#     inds = set(np.where(cur_grad > mid[0])[0]) & set(np.where(cur_grad < mid[1])[0])
#     if len(inds):
#         ind_dic['middle'].append(rd.sample(inds,num)[0])
#     return ind_dic

def choose_sample(cur_grad, num=1, small=0.05, mid=(0.3,0.7), large=0.95):        # 选中区域内的全部样本
    ind_dic = {'small': [], 'middle': [], 'large': []}
    # ----- small grad
    inds = np.where(cur_grad < small)[0]
    if len(inds):
        ind_dic['small'] += list(inds)               # choose one sample to show
    # ----- large grad
    inds = np.where(cur_grad > large)[0]
    if len(inds):
        ind_dic['large'] += list(inds)
    # ----- middle grad
    inds = set(np.where(cur_grad > mid[0])[0]) & set(np.where(cur_grad < mid[1])[0])
    if len(inds):
        ind_dic['middle'] += list(inds)
    return ind_dic


def save_img(x,y,grad,path):
    x = x*128 + 127.5                           # x可能是被flipped过的，这个不管
    x_arr = x.cpu().numpy()
    y_arr = y.cpu().numpy()
    for i in range(x_arr.shape[0]):
        x = np.transpose(x_arr[i],(1,2,0))      # chw to hwc
        x = x[:,:,::-1]                         # rgb to bgr
        # timestr = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
        label_str = str(y_arr[i])
        g = grad[i]
        rand_num = str(rd.random())[2:6]
        cv2.imwrite(os.path.join(path, label_str+'_' + str(g) + '_'+ rand_num + '.jpg'), x)


def save_choosed_sample(cur_grad, X, Y, name):
    # -----
    path1 = os.path.join(name, 'small_grad')
    path2 = os.path.join(name, 'middle_grad')
    path3 = os.path.join(name, 'large_grad')
    if not os.path.exists(path1):
        os.makedirs(path1)
    if not os.path.exists(path2):
        os.makedirs(path2)
    if not os.path.exists(path3):
        os.makedirs(path3)

    ind_dic = choose_sample(cur_grad)
    # ----- save imgs of small grad
    if len(ind_dic['small']):
        x = X[ind_dic['small']]
        y = Y[ind_dic['small']]
        save_img(x,y,cur_grad[ind_dic['small']],path1)
    # ----- save imgs of mid grad
    if len(ind_dic['middle']):
        x = X[ind_dic['middle']]
        y = Y[ind_dic['middle']]
        save_img(x,y,cur_grad[ind_dic['middle']],path2)
    # ----- save imgs of large grad
    if len(ind_dic['large']):
        x = X[ind_dic['large']]
        y = Y[ind_dic['large']]
        save_img(x,y,cur_grad[ind_dic['large']],path3)


class CalcGrad(nn.Module):
    def __init__(self, device=None):
        super(CalcGrad, self).__init__()
        self.grad = []
        self.device_id = device

    def forward(self, input, target):
        logpt = F.softmax(input, dim=1)
        # logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target.unsqueeze(1)).view(-1)
        # logpt.data.exp()
        cur_grad = np.array(torch.abs(logpt.data - 1).cpu())
        self.grad += list(cur_grad)
        return cur_grad


def draw_and_save(cha_grad, name):    # (ori_grad, cha_grad, name):
    bins = np.linspace(0.0,1.0,100)
    # fig, axs = plt.subplots(1,2,figsize=(6,4))
    # n, bins1 = np.histogram(np.array(ori_grad),bins)  # density/normed为频率直方图
    # axs[0].plot(n/np.sum(n))                          # axs[0].hist()
    # axs[0].set_title('origin_grad')
    # axs[0].grid()
    # n, _ = np.histogram(np.array(cha_grad),bins)
    # axs[1].plot(n/np.sum(n))
    # axs[1].set_title('changed_grad')
    # axs[1].grid()
    # fig.suptitle('%s'%name)
    # plt.savefig('%s.jpg' % name)
    n, _ = np.histogram(np.array(cha_grad), bins)
    plt.plot(bins[1:], n/np.sum(n))
    plt.title('Gradient statistics')
    plt.xlabel('gradient norm')
    plt.ylabel('fraction of samples')
    plt.grid()
    plt.savefig('%s.jpg' % name)

    return 0


if __name__ == '__main__':
    # # ------------------------- 1
    # GPU_ID = [0]
    # DEVICE = torch.device("cuda:%d" % GPU_ID[0] if torch.cuda.is_available() else "cpu")
    #
    # DATA_ROOT = '/data/code/insightface/datasets/ms1m-retinaface-t1'
    # EVAL_ROOT = '/data/xk/face_evoLVe_PyTorch-master/eval_data'
    # BATCH_SIZE = 64
    # img_size = (112, 112)
    # embedding_size = 512        # ****** 1/4
    # sample = True
    #
    # backbone_name = 'IR_50'    #  'MobileFaceNet_air'    #           ****** 2/4
    # head_name = 'AirFace'      # 'CosFace'         ****** 3/4
    # checkpoint_path = './buffer_model/IR_50_AirFace_2019-11-01_06_58_24-done-BEST/checkpoint.tar'   # ****** 4/4
    # name = './experiments/2_GHM/' + checkpoint_path.split('/')[-2]
    # CBAM = False
    # tanh_act = False
    #
    # # ------------------------- 2
    # train_loader = get_train_loader_rec(DATA_ROOT, BATCH_SIZE, img_size)
    # num_class = train_loader.num_classes  # 93431, 85742
    # train_loader = mx.io.PrefetchingIter(train_loader)
    # # train_loader.reset()
    #
    # backbone = gen_backbone(backbone_name, img_size, CBAM, tanh_act)
    # head = gen_head(head_name, embedding_size, num_class, GPU_ID)
    # backbone,head = load_pth_model(checkpoint_path, backbone, head)
    # backbone = backbone.to(DEVICE)
    # head = head.to(DEVICE)
    #
    # # cal_grad_ori = CalcGrad()
    # cal_grad_cha = CalcGrad()
    # backbone.eval()
    # head.eval()
    #
    # # ------------------------- 3 on training data
    # for iii, db_data in tqdm(enumerate(train_loader)):
    #     if iii == 300:
    #         break
    #     inputs = torch.from_numpy(db_data.data[0].asnumpy())
    #     labels = torch.from_numpy(db_data.label[0].asnumpy())
    #     del db_data
    #
    #     inputs = inputs.to(DEVICE)
    #     labels = labels.to(DEVICE).long()
    #
    #     with torch.no_grad():
    #         features = backbone(inputs)
    #         outputs = head(features, labels)
    #
    #         cur_grad = cal_grad_cha(outputs[0],labels)
    #         # cal_grad_ori(outputs[1],labels)
    #         if sample:
    #             save_choosed_sample(cur_grad, inputs, labels, name)
    #
    # # ------------------------- 4
    # draw_and_save(cal_grad_cha.grad, name)
    # print('done')

    # ------------------------- 5  2019.11.17   将指定类的所有样本转为图片保存到指定位置
    label_ls = [40308,34549,73634]
    for label in label_ls:
        show_samples_of_one_class(label=label,
                                path_imgrec='/data/code/insightface/datasets/ms1m-retinaface-t1/train.rec',
                                path='./experiments/2_GHM/IR_50_AirFace_2019-11-01_06_58_24-done-BEST')
