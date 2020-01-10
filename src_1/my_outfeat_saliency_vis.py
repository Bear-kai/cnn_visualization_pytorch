import torch
from backbone.my_model_irse import IR_50
from src_1.my_utils import load_pth_model, read_imgs, saliency_imgs_vis, saliency_vis_square


def out_feat_sali_1C(backbone, imgs, device, idx=0):
    """ Compute saliency map for one specified channel of feature(vector), support several imgs
        input:
                backbone: pre-trained network body, output feature vector
                imgs: torch.tensor
                device: specify cpu or gpu
                idx: specify channel to compute saliency maps
        return:
                sali_maps: numpy.ndarray
    """
    flag_bk_save = backbone.training
    backbone.eval()
    imgs.requires_grad = True

    feat = backbone(imgs)
    val = feat[:,idx]
    val.backward(torch.ones(val.shape).to(device))
    sali_maps, _ = imgs.grad.data.abs().max(dim=1)

    backbone.zero_grad()
    backbone.train(flag_bk_save)
    imgs.grad.detach_()
    imgs.grad = None
    imgs.requires_grad = False

    return sali_maps.cpu().numpy()


def out_feat_sali_mergeC(backbone, imgs, device):
    """ Compute saliency map for merged channel of feature(vector), support several imgs.
        Equivalently, compute saliency map for each channel and sum the maps together.
        input:
                backbone: pre-trained network body, output feature vector
                imgs: torch.tensor
                device: specify cpu or gpu
        return:
                sali_maps: numpy.ndarray
    """
    flag_bk_save = backbone.training
    backbone.eval()
    imgs.requires_grad = True

    feat = backbone(imgs)
    feat.backward(torch.ones(feat.shape).to(device))
    sali_maps, _ = imgs.grad.data.abs().max(dim=1)

    backbone.zero_grad()
    backbone.train(flag_bk_save)
    imgs.grad.detach_()
    imgs.grad = None
    imgs.requires_grad = False

    return sali_maps.cpu().numpy()


def out_feat_sali_NC_1samp(backbone,input_img,N=9):
    """ Compute saliency map for N specified channel of feature(vector), support just one img
        input:
                backbone: pre-trained network body, output feature vector
                input_img: torch.tensor, 1*C*H*W
                N: specify number of channels
        return:
                sali_maps: numpy.ndarray
    """
    img = input_img.clone()
    if len(img.shape)==4 and img.shape[0]>1:
        img = img[0].unsqueeze(0)
    if len(img.shape)==3:
        img = img.unsqueeze(0)
    flag_bk_save = backbone.training
    backbone.eval()
    img.requires_grad = True

    feat = backbone(img)
    if N > feat.shape[1]:
        raise ValueError('Param N should be smaller than feature dim %d'%feat.shape[1])
    sali_maps = []

    for i in range(N):
        feat[0,i].backward(retain_graph=True)
        saliency_map,_ = img.grad.data.abs().max(dim=1)
        sali_maps.append(saliency_map)
        backbone.zero_grad()
        img.grad.detach_()
        img.grad = None
    sali_maps = torch.cat(sali_maps,dim=0)

    backbone.zero_grad()
    backbone.train(flag_bk_save)
    img.requires_grad = False

    return sali_maps.cpu().numpy()


def main():
    # param setting
    gpu_id = [1]               # set None to use cpu ; set [0] to use gpu_0
    device = torch.device('cpu')
    if gpu_id is not None:
        device = torch.device('cuda:%d' % gpu_id[0])
    checkpoint_path = '../buffer_model/IR_50_AirFace/checkpoint.tar'
    save_path = '../results/outfeat_sali_vis'

    # define model
    backbone = IR_50(input_size=(112, 112), opt='E')
    backbone, head = load_pth_model(checkpoint_path, backbone)
    backbone = backbone.to(device)

    # prepare inputs
    path_ls = ['../face_imgs/3.jpg', '../face_imgs/31.jpg', '../face_imgs/8.jpg']
    labels = torch.LongTensor([34340,51979,33861])
    imgs = read_imgs(path_ls)
    labels = labels.to(device)
    imgs = imgs.to(device)

    # compute out feature saliency maps
    sali_maps = out_feat_sali_1C(backbone, imgs, device, idx=0)
    saliency_vis_square(sali_maps, save_path=save_path, file_name='sali_1C_square.jpg')
    saliency_imgs_vis(imgs, labels, sali_maps, save_path=save_path, file_name='sali_1C.jpg')

    sali_maps = out_feat_sali_mergeC(backbone, imgs, device)
    saliency_vis_square(sali_maps, save_path=save_path, file_name='sali_mergeC_square.jpg')
    saliency_imgs_vis(imgs, labels, sali_maps, save_path=save_path, file_name='sali_mergeC.jpg')

    sali_maps = out_feat_sali_NC_1samp(backbone, imgs[0], N=9)
    saliency_vis_square(sali_maps, save_path=save_path, file_name='sali_NC_square.jpg')
    print('done')


if __name__ == '__main__':
    main()
