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
from tqdm import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from robust_analysis_3_1 import gen_backbone, load_pth_model, load_pth_model_pub, get_train_loader_rec, cal_feat_norm


count_conv = 0
count_bn = 0
save_path = ''
def draw_and_save(data_pool,feat_norm, num, size, mtype):
    # data_pool: C*(NHW)
    # feat_norm: C
    global count_conv, count_bn, save_path
    count = 0
    if mtype == nn.Conv2d:
        mtype = 'conv'
        count_conv += 1
        count = count_conv
    elif mtype == nn.BatchNorm2d:
        mtype = 'bn'
        count_bn += 1
        count = count_bn
    rows = 3
    cols = 3
    figsize = (9, 9)
    fig1, axs = plt.subplots(rows, cols, figsize=figsize)
    idx = 0
    stop = 0
    for i in range(rows):
        for j in range(cols):
            if i==0 and j==0:
                feat_norm = feat_norm.cpu().numpy()
                _, bins, _ = axs[0,0].hist(feat_norm, bins=10, normed=1)
                axs[0,0].grid()
                axs[0,0].set_xlabel(r'FeatNorm $\mu$=%.2f $\sigma$=%.2f'%(np.mean(feat_norm), np.std(feat_norm)))
            else:
                if stop:
                    break   # skip out inner loop
                else:
                    tmp = data_pool[idx,:].cpu().numpy()
                    n, bins, patches = axs[i,j].hist(tmp, bins=100, normed=1)
                    mu = np.mean(tmp)
                    sigma = np.std(tmp)
                    y = mlab.normpdf(bins, mu, sigma)  # 拟合一条最佳正态分布曲线y
                    axs[i,j].plot(bins, y, 'r--')
                    axs[i,j].set_xlabel('channel_%d_norm_%.2f'%(idx+1, feat_norm[idx]))
                    # axs[i, j].set_ylabel('probability')
                    axs[i,j].grid()
                    axs[i,j].text(x=bins.min(), y=n.max(), s=r'$\mu$=%.2f, $\sigma$=%.2f'%(mu, sigma), fontsize=12)
                    idx += 1
                    if idx == num:
                        stop = 1
    # plt.suptitle('%s_%d'%(mtype, count))
    plt.tight_layout()

    file_name = '%s_%d_%s.jpg'%(mtype, count, size)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path,file_name))
    plt.close()


def statistic(m, x, y):
    num = 8
    if num > y.shape[1]:
        num = y.shape[1]
    if y.shape[2]==y.shape[3]==1:           #  CBAM attention
        print('skip 1x1 feature map')
        return
    feat_norm = y.data.norm(dim=(2,3)).mean(dim=0)                          # NCHW: channel norm across samples
    data_pool = y.data[:,:num,:,:].clone().transpose(0,1).reshape(num,-1)   # 选取前num个通道进行取值统计
    draw_and_save(data_pool,feat_norm, num, str(list(m.weight.shape)), type(m))


register_hooks = {
    nn.Conv2d: statistic,
    nn.BatchNorm2d: statistic}


def profile(model, inputs, verbose=True):
    handler_collection = []

    def add_hooks(m):
        if len(list(m.children())) > 0:
            return
        m_type = type(m)          # {type}<class 'torch.nn.modules.conv.Conv2d'>，等价于register_hooks中的Conv2d
        fn = None
        if m_type in register_hooks:
            fn = register_hooks[m_type]
        if fn is not None and verbose:
            print("Register counter for module %s" % str(m))
            handler = m.register_forward_hook(fn)
            handler_collection.append(handler)

    flag_save = model.training
    model.eval()
    model.apply(add_hooks)
    with torch.no_grad():
        model(inputs)
    model.train(flag_save)   # reset model to original status

    for handler in handler_collection:
        handler.remove()

    return 1


if __name__ == '__main__':
    # ------------------------- 1
    GPU_ID = [0]
    DEVICE = torch.device("cuda:%d" % GPU_ID[0] if torch.cuda.is_available() else "cpu")

    DATA_ROOT = '/data/code/insightface/datasets/ms1m-retinaface-t1'
    BATCH_SIZE = 128
    img_size = (112, 112)
    embedding_size = 128      # ****** 1/3

    backbone_name = 'MobileFaceNet_air'    # 'MobileFaceNet_air'  #  IR_50         ****** 2/3
    checkpoint_path = './buffer_model/MobileFaceNet_air_CBAM_AirFace_2019-10-14_02_56_46-done/checkpoint.tar'   # ****** 3/3
    # checkpoint_path = './model_zoo/ir50_private_asia/backbone_ir50_private_asia.pth'
    save_path = './experiments/3_2_featNorm/' + checkpoint_path.split('/')[-2]
    CBAM = True        # ******
    tanh_act = True    # ******

    # ------------------------- 2
    backbone = gen_backbone(backbone_name, img_size, CBAM, tanh_act)
    if backbone_name == 'IR_50_pub':
        backbone = load_pth_model_pub(checkpoint_path, backbone)
    else:
        backbone = load_pth_model(checkpoint_path, backbone)

    backbone = backbone.to(DEVICE)
    backbone.eval()

    # ------------------------- 3.1 on training data
    train_loader = get_train_loader_rec(DATA_ROOT, BATCH_SIZE, img_size)
    train_loader = mx.io.PrefetchingIter(train_loader)
    # train_loader.reset()
    norm_ls = []
    for iii, db_data in tqdm(enumerate(train_loader)):
        if iii == 1:
            break
        inputs = torch.from_numpy(db_data.data[0].asnumpy())
        del db_data
        inputs = inputs.to(DEVICE)
        profile(backbone, inputs, verbose=True)

    print('done')
