import torch
import torch.nn as nn
import torch.optim as optim
from backbone.my_model_irse import IR_50, Flatten
from src_1.my_utils import load_pth_model, read_imgs, saliency_vis_square, postprocess


def mid_feat_saliency_profile(backbone, imgs, device, mergeC=True):
    """ Compute siliency maps for the middle feature maps just before down-sample.
        input:
                backbone: pre-trained network body, output feature vector
                imgs: torch.tensor
                device: specify cpu or gpu
                mergeC: True--Compute saliency map for merged channel of feature maps
                last_map_size:
        output:
                sali_maps_ls: list of saliency maps corresponding to features of middle layers,
                              each element is [N,imgH,imgW], where N is the sample of imgs.
                map_size_ls : list of feature_map_size
    """
    handler_collection = []
    feat_map_pre_dw = []    # feature map before down-sample
    sali_maps_ls = []
    map_size_ls = []
    register_hooks = [nn.Conv2d, nn.MaxPool2d, Flatten, nn.AdaptiveAvgPool2d]  # can be changed accordingly

    def record_feat_map_pre_dw(m, x, y):
        """ m: module, x: tuple, y: tensor """
        save_flag = 0
        if type(m) in [nn.Conv2d, nn.MaxPool2d]:
            # Down-sample case
            if x[0].shape[2] != y.shape[2]:
                save_flag = 1
        elif type(m) in [Flatten, nn.AdaptiveAvgPool2d]:
            save_flag = 1

        if save_flag:
            feat_map_pre_dw.append(x[0].clone())
            map_size_ls.append(x[0].shape[2])


    def add_hooks(m):
        if len(list(m.children())) > 0:
            return
        m_type = type(m)          # {type}<class 'torch.nn.modules.conv.Conv2d'> equals to nn.Conv2d
        fn = None
        if m_type in register_hooks:
            fn = record_feat_map_pre_dw
        if fn is not None:
            print("Register hook for module %s" % str(m))
            handler = m.register_forward_hook(fn)
            handler_collection.append(handler)

    flag_save = backbone.training
    backbone.eval()
    backbone.apply(add_hooks)
    imgs.requires_grad = True
    optimizer_1 = optim.SGD(backbone.parameters(), lr=1e-6)
    optimizer_2 = optim.SGD([imgs], lr=1e-6)
    backbone(imgs)

    for feat_map in feat_map_pre_dw:
        optimizer_1.zero_grad()
        optimizer_2.zero_grad()
        if mergeC:
            feat_map.backward(torch.ones(feat_map.shape).to(device), retain_graph=True)
            sali_maps, _ = imgs.grad.data.abs().max(dim=1)
        else:
            val = feat_map[:,0,:,:]     # default the first channel
            val.backward(torch.ones(val.shape).to(device), retain_graph=True)
            sali_maps, _ = imgs.grad.data.abs().max(dim=1)
        sali_maps_ls.append(sali_maps.cpu())

    # reset model to original status and clear handler
    backbone.train(flag_save)
    imgs.grad.detach_()
    imgs.grad = None
    imgs.requires_grad = False
    for handler in handler_collection:
        handler.remove()

    return sali_maps_ls, map_size_ls


def wrap_profile_1stC(backbone, imgs, device, save_path=''):
    """ wrapper of functions mid_feat_saliency_profile() and saliency_vis_square()
        input:
                backbone: pre-trained network body, output feature vector
                imgs: torch.tensor
                device: specify cpu or gpu
                save_path: path to save figures
    """
    sali_maps_ls, map_size_ls = mid_feat_saliency_profile(backbone, imgs, device, mergeC=False)
    for i,sali_maps in enumerate(sali_maps_ls):
        saliency_vis_square(sali_maps, save_path=save_path, file_name='vis1C_size-%d_layer-%d.jpg'%(map_size_ls[i],i))
    # after '.cpu()', modify imgs_arr will not affect imgs
    imgs_arr = imgs.data.cpu().numpy().transpose(0, 2, 3, 1)
    saliency_vis_square(postprocess(imgs_arr), save_path=save_path, file_name='orig_img.jpg')


def wrap_profile_mergeC(backbone, imgs, device, save_path=''):
    """ wrapper of functions mid_feat_saliency_profile() and saliency_vis_square()
        input:
                backbone: pre-trained network body, output feature vector
                imgs: torch.tensor
                device: specify cpu or gpu
                save_path: path to save figures
        """
    sali_maps_ls, map_size_ls = mid_feat_saliency_profile(backbone, imgs, device, mergeC=True)
    for i,sali_maps in enumerate(sali_maps_ls):
        saliency_vis_square(sali_maps, save_path=save_path, file_name='visMC_size-%d_layer-%d.jpg'%(map_size_ls[i],i))
    # after '.cpu()', modify imgs_arr will not affect imgs
    imgs_arr = imgs.data.cpu().numpy().transpose(0, 2, 3, 1)
    saliency_vis_square(postprocess(imgs_arr), save_path=save_path, file_name='orig_img.jpg')


def main_1():
    # param setting
    gpu_id = [1]               # set None to use cpu ; set [0] to use gpu_0
    device = torch.device('cpu')
    if gpu_id is not None:
        device = torch.device('cuda:%d' % gpu_id[0])
    checkpoint_path = '../buffer_model/IR_50_AirFace/checkpoint.tar'
    save_path = '../results/midfeat_sali_vis_ir50'

    # define model
    backbone = IR_50(input_size=(112, 112), opt='E')
    backbone, head = load_pth_model(checkpoint_path, backbone)
    backbone = backbone.to(device)

    # prepare inputs
    path_ls = ['../face_imgs/3.jpg', '../face_imgs/31.jpg', '../face_imgs/8.jpg']
    imgs = read_imgs(path_ls)
    imgs = imgs.to(device)

    # compute middle feature saliency maps
    wrap_profile_1stC(backbone, imgs, device, save_path=save_path)
    wrap_profile_mergeC(backbone, imgs, device, save_path=save_path)
    print('done')


def main_2():
    # param setting
    gpu_id = [1]  # set None to use cpu ; set [0] to use gpu_0
    device = torch.device('cpu')
    if gpu_id is not None:
        device = torch.device('cuda:%d' % gpu_id[0])
    save_path = '../results/midfeat_sali_vis_alexnet'

    # define model
    from torchvision import models
    alex_model = models.alexnet(pretrained=True)
    alex_model = alex_model.to(device)

    # prepare inputs
    path_ls = ['../input_images/snake.jpg', '../input_images/cat_dog.png', '../input_images/spider.png']
    imgs = read_imgs(path_ls,size=(224,224))
    imgs = imgs.to(device)

    # compute middle feature saliency maps
    wrap_profile_1stC(alex_model, imgs, device, save_path=save_path)
    wrap_profile_mergeC(alex_model, imgs, device, save_path=save_path)
    print('done')


if __name__ == '__main__':
    # main_1()
    main_2()
