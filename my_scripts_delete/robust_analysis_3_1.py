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
import torchvision.transforms as transforms
import numpy as np
import os
from backbone.my_model_irse import IR_50
from backbone.model_irse import IR_50_pub
from backbone.MobileFaceNet import MobileFaceNet, MobileFaceNet_air
from util.image_iter_rec import FaceImageIter
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from tqdm import tqdm
import cv2
from util.utils import get_val_data


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


def load_pth_model(checkpoint_path, backbone):
    print("=" * 60)
    if os.path.isfile(checkpoint_path):
        print("loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        backbone.load_state_dict(checkpoint['backbone'])
    else:
        print("No Checkpoint exists at '{}'. Train from Scratch".format(checkpoint_path))
    print("=" * 60)
    return backbone


def load_pth_model_pub(checkpoint_path, backbone):
    print("=" * 60)
    if os.path.isfile(checkpoint_path):
        print("loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        backbone.load_state_dict(checkpoint)        # ***
    else:
        print("No Checkpoint exists at '{}'".format(checkpoint_path))
    print("=" * 60)
    return backbone


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


def get_embedding_val(backbone, carray, device, embedding_size=128, batch_size=64):
    idx = 0
    embeddings = torch.zeros([len(carray), embedding_size],dtype=torch.float32)
    with torch.no_grad():
        while idx + batch_size <= len(carray):
            batch = torch.tensor(carray[idx:idx + batch_size][:, [2, 1, 0], :, :])  # bgr转rgb
            embeddings[idx:idx + batch_size] = backbone(batch.to(device)).cpu()     # 可以直接将tensor赋值给numpy ndarray，但反过来不行！
            idx += batch_size
        if idx < len(carray):       # 对于最后一个batch
            batch = torch.tensor(carray[idx:])
            embeddings[idx:] = backbone(batch.to(device)).cpu()

    return embeddings


ccrop = transforms.Compose([ transforms.ToTensor(),
                             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
def get_embedding_test(backbone, buffer, img_size, device):
    input_blob = np.zeros((len(buffer), 3, img_size[0], img_size[1]),dtype=np.uint8)
    idx = 0
    for item in buffer:
        img = cv2.imread(item)[:, :, ::-1]      # to rgb
        img = np.transpose(img, (2, 0, 1))      # to c*h*w
        _img = np.copy(img)
        input_blob[idx] = _img
        idx += 1

    def ccrop_batch(input_batch):
        ccropped_imgs = torch.zeros(input_batch.shape)
        input_batch = np.transpose(input_batch, (0, 2, 3, 1))  # ==> NHWC
        for i, img in enumerate(input_batch):
            ccropped_imgs[i] = ccrop(img)
        return ccropped_imgs

    input_blob = ccrop_batch(input_blob)
    with torch.no_grad():  # 前向传播提取特征，不进行梯度计算
        embeddings = backbone(input_blob.to(device)).cpu()

    return embeddings


def cal_feat_norm(feats):
    return list(feats.norm(dim=1).numpy())


def draw_and_save(norm_ls, name):
    # bins = np.linspace(np.min(norm_ls), np.max(norm_ls), 100)
    # n, _ = np.histogram(np.array(norm_ls), bins)
    # plt.plot(bins[1:], n/np.sum(n))
    n, bins, patches = plt.hist(norm_ls, bins=100, normed=1)
    mu = np.mean(norm_ls)
    sigma = np.std(norm_ls)
    y = mlab.normpdf(bins, mu, sigma)  # 拟合一条最佳正态分布曲线y
    plt.plot(bins,y,'r--')
    plt.title('FeatureNorm statistics')
    plt.xlabel('feature norm')
    plt.ylabel('probability')
    plt.text(x=bins.min(),y=n.max(),s=r'$\mu$=%.3f, $\sigma$=%.3f'%(mu,sigma),fontsize=15)
    plt.grid()
    plt.savefig('%s.jpg' % name)
    plt.close()


if __name__ == '__main__':
    # ------------------------- 1
    GPU_ID = [0]
    DEVICE = torch.device("cuda:%d" % GPU_ID[0] if torch.cuda.is_available() else "cpu")

    DATA_ROOT = '/data/code/insightface/datasets/ms1m-retinaface-t1'
    EVAL_ROOT = '/data/xk/face_evoLVe_PyTorch-master/eval_data'
    BATCH_SIZE = 64
    img_size = (112, 112)
    embedding_size = 128        # ****** 1/3

    backbone_name = 'MobileFaceNet_air'    # 'MobileFaceNet_air'  #  IR_50         ****** 2/3
    checkpoint_path = './buffer_model/MobileFaceNet_air_CBAM_AirFace_2019-10-14_02_56_46-done/checkpoint.tar'   # ****** 3/3
    # checkpoint_path = './model_zoo/ir50_ms1m_epoch120/backbone_ir50_ms1m_epoch120.pth'
    name = './experiments/3_1_featNorm/' + checkpoint_path.split('/')[-2]
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
    # train_loader = get_train_loader_rec(DATA_ROOT, BATCH_SIZE, img_size)
    # train_loader = mx.io.PrefetchingIter(train_loader)
    # # train_loader.reset()
    # norm_ls = []
    # for iii, db_data in tqdm(enumerate(train_loader)):
    #     if iii == 1000:
    #         break
    #     inputs = torch.from_numpy(db_data.data[0].asnumpy())
    #     del db_data
    #     inputs = inputs.to(DEVICE)
    #     with torch.no_grad():
    #         feats = backbone(inputs)
    #         norm_ls += cal_feat_norm(feats.cpu())
    # draw_and_save(norm_ls, name)

    # ------------------------- 3.2 on valdiation data
    # lfw, cfp_ff, cfp_fp, agedb, calfw, cplfw, vgg2_fp, _, _, _, _, _, _, _ = get_val_data(EVAL_ROOT)
    # data_ls = [lfw, cfp_ff, cfp_fp, agedb, calfw, cplfw, vgg2_fp]
    # data_name = ['lfw','cfp_ff','cfp_fp','agedb','calfw','cplfw','vgg2_fp']
    # for i, dat in enumerate(data_ls): # [lfw,cfp_ff,cfp_fp,agedb,calfw,cplfw,vgg2_fp]
    #     print('processing the %d-th dataset'%(i+1))
    #     feats = get_embedding_val(backbone,dat,DEVICE,embedding_size,BATCH_SIZE)
    #     norm_ls = cal_feat_norm(feats)
    #     draw_and_save(norm_ls, name+'-%s'%data_name[i])

    # ------------------------- 3.3 on testing data
    path1 = '/data_4t/xk/datasets/MegaFace_test_data/'
    facescrub_lst = os.path.join(path1, 'facescrub_lst_3530')
    megaface_lst = os.path.join(path1, 'megaface_lst_1027058')
    facescrub_root = os.path.join(path1, 'megaface_testpack_v1.0/facescrub_images_N3530')
    megaface_root = os.path.join(path1, 'megaface_testpack_v1.0/megaface_images_N1027058')

    print('extract features of facescrub\n')
    i = 0
    buffer = []
    norm_ls = []
    for line in open(facescrub_lst, 'r'):  # line: 'Christian_Bale/Christian_Bale_11936.png\n'
        if i % 1000 == 0:
            print("writing facescrub %d" % i)
        i += 1
        image_path = line.strip()
        _path = image_path.split('/')
        a, b = _path[-2], _path[-1]         # a=Christian_Bale, b=Christian_Bale_11936.png
        image_path = os.path.join(facescrub_root, image_path)
        if not os.path.isfile(image_path):
            print('image not exists: ', image_path)
            continue
        buffer.append(image_path)
        if len(buffer) == BATCH_SIZE:
            feats = get_embedding_test(backbone, buffer, img_size, DEVICE)
            norm_ls += cal_feat_norm(feats.cpu())
            buffer = []
    if len(buffer) > 0:
        embeddings = get_embedding_test(backbone, buffer, img_size, DEVICE)
        norm_ls += cal_feat_norm(feats.cpu())
        del buffer
    draw_and_save(norm_ls, name)
    print('fs stat %d' %i )

    print('extract features of megaface\n')
    i = 0
    buffer = []
    norm_ls = []
    for line in open(megaface_lst, 'r'):
        if i % 1000 == 0:
            print("writing megaface %d" % i)
        if i >= (BATCH_SIZE*1000):
            break
        i += 1
        image_path = line.strip()
        _path = image_path.split('/')
        a1, a2, b = _path[-3], _path[-2], _path[-1]
        image_path = os.path.join(megaface_root, image_path)
        if not os.path.isfile(image_path):
            print('image not exists: ', image_path)
            continue
        buffer.append(image_path)
        if len(buffer) == BATCH_SIZE:
            embeddings = get_embedding_test(backbone, buffer, img_size, DEVICE)
            norm_ls += cal_feat_norm(feats.cpu())
            buffer = []
    draw_and_save(norm_ls, name)
    print('mf stat %d' % i)

    # ------------------------- 4
    print('done')
