# ====== 现象：
# 1. 无预处理时，大模型的测试集性能稍降低几个点，小模型直接崩（近似0）
# 2. 有预处理时，在验证集上表现接近，甚至小模型略优时，在测试集上大模型还是比小模型高好几个点（note by huhui: 可能是验证集难度不够，不足以在验证集上区分二者性能）
# ====== 初步结论：大模型的鲁棒性优于小模型
# ====== 鉴于以上两种现象，尤其是现象1，尝试从以下4个方面进行探索，以期望更深入地理解模型，继而优化模型。
# 1.统计各层权值范数; 2.给定一批样本，统计梯度信息(ref AAAI19-GHM);
# 3.给定一批样本，3.1统计输出特征范数(ref ICCV19-LargeNorm)， 3.2 统计一个batch的中间特征取值分布&范数;
# 4.特征可视化(ref ECCV14 + ArXiv14)
# ref: https://github.com/huanghao-code/VisCNN_ICLR_2014_Saliency/blob/master/demo.py
# https://github.com/artvandelay/Deep_Inside_Convolutional_Networks

# 缺乏对输入图像的预处理，待加！

import torch
import cv2
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import os
import matplotlib.pyplot as plt
from robust_analysis_3_1 import gen_backbone
from robust_analysis_2 import gen_head, load_pth_model


def class_saliency_vis(backbone,head,imgs,labels,posterior=False):    # image-specific, support multi-imgs
    imgs.requires_grad = True
    feat = backbone(imgs)
    logits_mod, logits_orig = head(feat,labels)
    if posterior:
        loss_func = torch.nn.CrossEntropyLoss()
        loss = loss_func(logits_orig, labels)
        loss.backward(torch.ones(loss.shape).to(DEVICE))
    else:
        score = logits_orig.gather(1, labels.view(-1, 1)).squeeze()
        score.backward(torch.ones(score.shape).to(DEVICE))
    saliency_maps, idx = imgs.grad.data.abs().max(dim=1)
    imgs.requires_grad = False
    return saliency_maps.cpu()


def profile(model, inputs, verbose=True, mergeC=True, idx=0, last_map_size=7):
    handler_collection = []
    feat_map_pre_s2 = []
    map_size_ls = []
    saliency_maps_ls = []   # each element is [N,imgH,imgW], #elements is #stages/layers
    last_map_size = 7       # save feat

    def record_feature_map_pre_s2(m, x, y):
        if x[0].shape[2] == y.shape[2]:         # x: tuple, y:tensor
            return                              # skip module within a block/layer
        else:
            feat_map_pre_s2.append(x[0].clone()) # correspond to downsample case: save the last conv module output of each layer
            map_size_ls.append(x[0].shape[2])
            if y.shape[2] == last_map_size:      # here save the 1st conv module output of the last layer
                feat_map_pre_s2.append(y.clone())
                map_size_ls.append(last_map_size)

    def add_hooks(m):
        if len(list(m.children())) > 0:
            return
        m_type = type(m)          # {type}<class 'torch.nn.modules.conv.Conv2d'>，等价于register_hooks中的Conv2d
        fn = None
        if m_type in [torch.nn.Conv2d]:
            fn = record_feature_map_pre_s2
        if fn is not None and verbose:
            print("Register counter for module %s" % str(m))
            handler = m.register_forward_hook(fn)
            handler_collection.append(handler)

    flag_save = model.training
    model.eval()
    model.apply(add_hooks)
    optimizer_1 = optim.SGD(model.parameters(), lr=1e-6)
    optimizer_2 = optim.SGD([inputs], lr=1e-6)
    model(inputs)
    for feat_map in feat_map_pre_s2:
        optimizer_1.zero_grad()
        optimizer_2.zero_grad()
        if mergeC:
            feat_map.backward(torch.ones(feat_map.shape).to(DEVICE), retain_graph=True)  # img某个位置的grad是feat各元素对img该位置的梯度之和
            saliency_maps, _ = inputs.grad.data.abs().max(dim=1)
        else:
            val = feat_map[:,idx,:,:]
            val.backward(torch.ones(val.shape).to(DEVICE), retain_graph=True)
            saliency_maps, _ = inputs.grad.data.abs().max(dim=1)
        saliency_maps_ls.append(saliency_maps.cpu())

    model.train(flag_save)   # reset model to original status
    for handler in handler_collection:
        handler.remove()

    return saliency_maps_ls, map_size_ls

# show 1 channel, supporting multiple imgs
def mid_feature_saliency_vis_1C(backbone, imgs, idx=0, save_root=''):
    imgs.requires_grad = True
    saliency_maps_ls, map_size_ls = profile(backbone, imgs, verbose=True, mergeC=False, idx=idx)
    for i,saliency_maps in enumerate(saliency_maps_ls):
        saliency_vis_square(saliency_maps, padsize=5, padval=0, save_path=os.path.join(save_root,'vis1C_size-%d_layer-%d.jpg'%(map_size_ls[i],i)))
    imgs = imgs.cpu().data.numpy().transpose(0, 2, 3, 1)
    saliency_vis_square(postprocess(imgs), padsize=5, padval=0, save_path=os.path.join(save_root,'orig_img.jpg'))


# show merged channel (the accumulated gradient), supporting multiple imgs
def mid_feature_saliency_vis_mergeC(backbone, imgs, save_root=''):
    imgs.requires_grad = True
    saliency_maps_ls, map_size_ls = profile(backbone, imgs, verbose=True, mergeC=True)
    for i,saliency_maps in enumerate(saliency_maps_ls):
        saliency_vis_square(saliency_maps, padsize=5, padval=0, save_path=os.path.join(save_root,'visMC_size-%d_layer-%d.jpg'%(map_size_ls[i],i)))
    imgs = imgs.cpu().data.numpy().transpose(0, 2, 3, 1)
    saliency_vis_square(postprocess(imgs), padsize=5, padval=0, save_path=os.path.join(save_root, 'orig_img.jpg'))


# show 1 channel, supporting multiple imgs # 多张图片的输出特征的第一个通道
def out_feature_saliency_vis_1C(backbone, imgs, idx=0):
    imgs.requires_grad = True
    feat = backbone(imgs)
    val = feat[:,idx]
    val.backward(torch.ones(val.shape).to(DEVICE))
    saliency_maps, _ = imgs.grad.data.abs().max(dim=1)
    imgs.requires_grad = False
    return saliency_maps.cpu()


# show merged channel (the accumulated gradient), supporting multiple imgs # 多张图片的输出特征的所有通道
def out_feature_saliency_vis_mergeC(backbone, imgs):
    imgs.requires_grad = True
    feat = backbone(imgs)
    feat.backward(torch.ones(feat.shape).to(DEVICE))    # img某个位置的grad是feat各元素对img该位置的梯度之和
    saliency_maps, _ = imgs.grad.data.abs().max(dim=1)
    imgs.requires_grad = False
    return saliency_maps.cpu()


# show the first N channels of one sample   # 一张图片的前9个通道
def out_feature_saliency_vis_NC_1sample(backbone,img,N=9):
    if len(img.shape)==4 and img.shape[0]>1:
        img = img[0].unsqueeze(0)
    if len(img.shape)==3:
        img = img.unsqueeze(0)
    img.requires_grad = True
    feat = backbone(img)
    saliency_maps = []
    for i in range(N):
        feat[0,i].backward(retain_graph=True)
        saliency_map,_ = img.grad.data.abs().max(dim=1)
        saliency_maps.append(saliency_map)
        img.grad = None
    saliency_maps = torch.cat(saliency_maps,dim=0)
    img.requires_grad = False
    return saliency_maps.cpu()


def class_model_vis(backbone,head,x ,labels,lam=0.1,epoch=90,lr=0.01,posterior=False,save_path='',p=2,stages=[150,300]): # generate_img
    if epoch%9:
        raise ValueError('Set epoch=9*n where n is 1,2,3,4...')
    div_num = epoch//9
    num = x.shape[0]
    if not x.requires_grad:
        x.requires_grad = True
    alter_name = 'logits'
    if posterior:
        loss_func = torch.nn.CrossEntropyLoss()
        alter_name = 'loss'
    if save_path:
        fig, axs = plt.subplots(3, 3, figsize=(9, 9))
    optimizer_1 = optim.SGD([{'params': backbone.parameters()}, {'params': head.parameters()}], lr=1e-6)
    optimizer_2 = optim.SGD([x],lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_2, milestones=[*stages], gamma=0.1)
    val_ls = []
    val_ls_1 = []
    xnm_ls = []
    for i in range(epoch):
        feat = backbone(x)
        logits_mod, logits_orig = head(feat, labels)
        if posterior:
            loss = loss_func(logits_mod, labels)
            xnm = x.view(num,-1).norm(p=p,dim=1)
            if not loss.shape:
                loss = loss.unsqueeze(0)
            alter_val = loss[0].item()          # .detach().cpu()
            loss += lam * xnm                   # 两者均尽量小
            optimizer_1.zero_grad()
            optimizer_2.zero_grad()
            loss.backward(torch.ones(loss.shape).to(DEVICE))     # 支持非标量backward
            # x.data -= lr * x.grad.data
            optimizer_2.step()
            lr_scheduler.step()
            if loss.shape:
                alter_val_1 = loss[0].item()     # .detach().cpu()
            else:
                alter_val_1 = loss.item()
        else:
            score = logits_mod.gather(1, labels.view(-1, 1)).squeeze()
            xnm = x.view(num,-1).norm(p=p,dim=1)
            if not score.shape:
                score = score.unsqueeze(0)
            alter_val = score[0].item()
            score -= lam * xnm
            score = -1.0 * score
            optimizer_1.zero_grad()
            optimizer_2.zero_grad()
            score.backward(torch.ones(score.shape).to(DEVICE))
            # x.data -= lr * x.grad.data
            optimizer_2.step()
            lr_scheduler.step()
            if score.shape:
                alter_val_1 = score[0].item()     # .detach().cpu()
            else:
                alter_val_1 = score.item()
        if save_path:     # save imgs of the training process
            if (i+1)%div_num==0:
                ind = (i+1)//div_num - 1
                axs[ind//3,ind%3].imshow(postprocess(x[0].data.cpu().numpy().transpose(1,2,0)))
                axs[ind // 3, ind % 3].axis('off')
                axs[ind // 3, ind % 3].set_title('iter=%d, %s=%.3f'%(i+1,alter_name,alter_val))
                if (i+1)//div_num==9:
                    plt.tight_layout()
                    plt.savefig(save_path)
                    plt.close()

        val_ls.append(alter_val)
        val_ls_1.append(alter_val_1)
        xnm_ls.append(xnm)
    l1, = plt.plot(val_ls)
    l2, = plt.plot(val_ls_1)
    l3, = plt.plot(xnm_ls)
    plt.legend(handles=[l1,l2,l3,], labels=['%s'%alter_name, 'Total', 'norm'], loc='best')
    plt.savefig(save_path[:-4]+'_%s.jpg'%alter_name)
    plt.close()

    return x.cpu()


def saliency_vis_square(data, padsize=1, padval=0, save_path=''):   # support NHW and NHWC
    # data -= data.min()
    # data /= data.max()

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    fig = plt.figure(figsize=(9,9))
    plt.imshow(data, cmap=plt.cm.hot)  # gray
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()


def saliency_imgs_vis(images, labels, saliency_maps, save_path=''):
    N = images.shape[0]     # number of images
    if N==1:
        fig, axs = plt.subplots(2, N, figsize=(3, 6))
        axs[0].imshow(postprocess(images[0].data.numpy().transpose(1,2,0)))
        axs[0].axis('off')
        axs[0].set_title('label-%d'%labels[0])
        axs[1].imshow(postprocess(saliency_maps[0].numpy()), cmap=plt.cm.hot)   # https://blog.csdn.net/qq_28485501/article/details/82656614
        axs[1].axis('off')
    elif N>1:
        fig, axs = plt.subplots(2, N, figsize=(N*3, 6))
        for i in range(N):
            axs[0,i].imshow(postprocess(images[i].data.numpy().transpose(1,2,0)))
            axs[0,i].axis('off')
            axs[0,i].set_title('label-%d' % labels[i])
            axs[1,i].imshow(postprocess(saliency_maps[i].numpy()), cmap = plt.cm.hot)
            axs[1,i].axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def postprocess(img):
    img = img * 0.5 + 0.5
    img = (img - img.min()) / (img.max() - img.min()) * 255
    img = img.astype(np.uint8)
    return img


def show_gen_imgs(images,labels, file_name='gen_img.jpg', save_path='./experiments'):
    N = images.shape[0]     # number of images
    if N==1:
        plt.imshow(postprocess(images[0].data.numpy().transpose(1,2,0)))
        plt.axis('off')
        plt.title('label-%d'%labels[0])
    # elif N>1:
    #     col = int(np.ceil(N / 2))
    #     fig, axs = plt.subplots(2, col, figsize=(N*3, 6))
    #     for i in range(N):
    #         axs[i//col,i%col].imshow(postprocess(images[i].data.numpy().transpose(1,2,0)))
    #         axs[i//col,i%col].axis('off')
    #         axs[i//col,i%col].set_title('label-%d' % labels[i])
    plt.tight_layout()
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, file_name))
    plt.close()


def read_imgs(path_ls):
    normalize = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize(size=(112,112)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    imgs = []
    for img_path in path_ls:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = normalize(img).unsqueeze(0)  # [1,3,112,112]
        imgs.append(img)
    imgs = torch.cat(imgs,dim=0)
    return imgs


def main_1():
    # ------------------------- 1
    num_class = 93431  # 85742
    img_size = (112, 112)
    embedding_size = 512  # ****** 1/4

    backbone_name = 'IR_50'  # 'MobileFaceNet_air'  #     #      ****** 2/4
    head_name = 'AirFace'  # 'CosFace'                         ****** 3/4
    checkpoint_path = './buffer_model/IR_50_AirFace_2019-11-01_06_58_24-done-BEST/checkpoint.tar'  # ****** 4/4
    # checkpoint_path = './model_zoo/ir50_private_asia/backbone_ir50_private_asia.pth'
    save_path = './experiments/4_featVis/' + checkpoint_path.split('/')[-2]
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    CBAM = False  # ******
    tanh_act = False  # ******

    idx = 3
    path_ls = ['./experiments/%d.jpg' % idx]  # 31:51979  3：34340  8:33861
    labels = torch.LongTensor([51979])
    img = read_imgs(path_ls)
    labels = labels.to(DEVICE)
    img = img.to(DEVICE)

    # ------------------------- 2
    backbone = gen_backbone(backbone_name, img_size, CBAM, tanh_act)
    head = gen_head(head_name, embedding_size, num_class, GPU_ID)
    backbone, head = load_pth_model(checkpoint_path, backbone, head)

    backbone = backbone.to(DEVICE)
    head = head.to(DEVICE)
    backbone.eval()
    head.eval()

    # ------------------------- 3 Class Saliency Visualization
    # saliency_maps1 = class_saliency_vis(backbone, head, img, labels, posterior=False)
    # saliency_imgs_vis(img.cpu(), labels, saliency_maps1, save_path=save_path + '/sali-logit.jpg')

    # saliency_maps2 = class_saliency_vis(backbone, head, img, labels, posterior=True)
    # saliency_imgs_vis(img.cpu(),labels, saliency_maps2, save_path=save_path + '/sali_loss.jpg')

    # ------------------------- 4 Class Model Visualization
    x = torch.zeros(1, 3, 112, 112).to(DEVICE)
    gen_imgs = class_model_vis(backbone, head, x, labels, lam=5, epoch=450,lr=0.1, posterior=False,save_path=save_path + '/genimg_logit_tune.jpg')
    # show_gen_imgs(gen_imgs,labels, file_name='%d_gen_img.jpg'%idx, save_path=save_path)

    x = torch.zeros(1, 3, 112, 112).to(DEVICE)
    gen_imgs = class_model_vis(backbone, head,x,labels,lam=5,epoch=450,lr=0.1, posterior=True,save_path=save_path+'/genimg_loss_tune.jpg')
    # show_gen_imgs(gen_imgs,labels, file_name='%d_gen_img_post.jpg'%idx, save_path=save_path)

    print('done')


def main_2():
    # ------------------------- 1
    num_class = 93431  # 85742
    img_size = (112, 112)
    embedding_size = 512  # ****** 1/4

    backbone_name =  'IR_50'   #  'MobileFaceNet_air'  #       ****** 2/4
    head_name = 'AirFace'  # 'CosFace'                         ****** 3/4
    checkpoint_path = './buffer_model/IR_50_AirFace_2019-11-01_06_58_24-done-BEST/checkpoint.tar'  # ****** 4/4
    # checkpoint_path = './model_zoo/ir50_private_asia/backbone_ir50_private_asia.pth'
    save_path = './experiments/4_featVis_1/' + checkpoint_path.split('/')[-2] + '-1'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    CBAM = False  # ******
    tanh_act = False  # ******

    path_ls = ['./experiments/31.jpg','./experiments/3.jpg','./experiments/8.jpg','./experiments/0.jpg']  # 31:51979  3：34340  8:33861
    labels = torch.LongTensor([51979,34340,33861,0])
    imgs = read_imgs(path_ls)
    # labels = labels.to(DEVICE)
    imgs = imgs.to(DEVICE)

    # ------------------------- 2
    backbone = gen_backbone(backbone_name, img_size, CBAM, tanh_act)
    head = gen_head(head_name, embedding_size, num_class, GPU_ID)
    backbone, _ = load_pth_model(checkpoint_path, backbone, head)
    backbone = backbone.to(DEVICE)
    backbone.eval()

    # ------------------------- 3 feature Saliency Visualization
    saliency_maps = out_feature_saliency_vis_1C(backbone, imgs, idx=0)
    saliency_vis_square(saliency_maps, padsize=5, padval=0, save_path=save_path + '/feat_sali_1C.jpg')
    saliency_imgs_vis(imgs.cpu(), labels, saliency_maps, save_path=save_path + '/feat_sali_img_1C.jpg')

    saliency_maps = out_feature_saliency_vis_mergeC(backbone, imgs)
    saliency_vis_square(saliency_maps, padsize=5, padval=0, save_path=save_path + '/feat_sali_mergeC.jpg')
    saliency_imgs_vis(imgs.cpu(), labels, saliency_maps, save_path=save_path + '/feat_sali_img_mergeC.jpg')

    saliency_maps = out_feature_saliency_vis_NC_1sample(backbone, imgs[0], N=9)
    saliency_vis_square(saliency_maps, padsize=5, padval=0, save_path=save_path + '/feat_sali_NC_1samp.jpg')

    print('done')


def main_3():
    # ------------------------- 1
    img_size = (112, 112)
    backbone_name = 'IR_50'   # 'MobileFaceNet_air'   #           ****** 1/2
    checkpoint_path = './buffer_model/IR_50_AirFace_2019-11-01_06_58_24-done-BEST/checkpoint.tar'  # ****** 2/2
    # checkpoint_path = './model_zoo/ir50_private_asia/backbone_ir50_private_asia.pth'
    save_path = './experiments/4_featVis_2/' + checkpoint_path.split('/')[-2] + '-1'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    CBAM = False        # ******
    tanh_act = False    # ******

    path_ls = ['./experiments/31.jpg','./experiments/3.jpg','./experiments/8.jpg','./experiments/0.jpg']  # 31:51979  3：34340  8:33861
    imgs = read_imgs(path_ls)
    imgs = imgs.to(DEVICE)

    # ------------------------- 2
    backbone = gen_backbone(backbone_name, img_size, CBAM, tanh_act)
    backbone, _ = load_pth_model(checkpoint_path, backbone)
    backbone = backbone.to(DEVICE)
    backbone.eval()

    # from tensorboardX import SummaryWriter
    # from torchvision import models
    # vgg16 = models.vgg16()
    # inputs = torch.randn((1,3,224,224))
    # writer = SummaryWriter(comment='IR50_1101',log_dir='runs/exp-1')
    # writer.add_graph(vgg16, (inputs,))
    # # AttributeError: 'torch._C.Value' object has no attribute 'uniqueName'
    # # 版本问题,torch/tensorboardX可能需要降版本，在laptop的virtual_env上能运行
    # writer.close()

    # import tensorwatch as tw      # 须在jupter notebook中运行，laptop上运行仍有问题
    # tw.draw_model(backbone,[1,3,112,112])

    # ------------------------- 3 feature Saliency Visualization
    # mid_feature_saliency_vis_1C(backbone, imgs, idx=0, save_root=save_path)
    # mid_feature_saliency_vis_mergeC(backbone, imgs, save_root=save_path)

    print('done')


if __name__ == '__main__':
    torch.backends.cudnn.deterministic = True
    GPU_ID = [0]
    DEVICE = torch.device("cuda:%d" % GPU_ID[0] if torch.cuda.is_available() else "cpu")
    # main_1()
    # main_2()
    main_3()



