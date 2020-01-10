import torch
from backbone.my_model_irse import IR_50
from head.airface import AirFace
from src_1.my_utils import load_pth_model, read_imgs, saliency_imgs_vis, saliency_vis_square


def class_saliency_vis(backbone,head,imgs,labels,device,posterior=False):
    """ compute image-specific saliency map, support multi-imgs
        input:
                backbone: pre-trained network body, output feature vector
                head: pre-trained network head, output class logits
                imgs: torch.tensor
                labels: torch.tensor
                device: specify cpu or gpu
                posterior: True--loss based optimization, False--logit based optimization
        output:
                sali_maps: numpy.ndarray
    """
    flag_bk_save = backbone.training
    flag_hd_save = head.training
    backbone.eval()
    head.eval()
    imgs.requires_grad = True

    feat = backbone(imgs)
    logits = head(feat,labels)
    if posterior:
        loss_func = torch.nn.CrossEntropyLoss()
        loss = loss_func(logits, labels)
        loss.backward(torch.ones(loss.shape).to(device))
    else:
        score = logits.gather(1, labels.view(-1, 1)).squeeze()
        score.backward(torch.ones(score.shape).to(device))
    sali_maps, _ = imgs.grad.data.abs().max(dim=1)

    backbone.zero_grad()
    head.zero_grad()
    backbone.train(flag_bk_save)
    head.train(flag_hd_save)
    imgs.grad.detach_()
    imgs.grad = None
    imgs.requires_grad = False

    return sali_maps.cpu().numpy()


def main():
    # param setting
    gpu_id = [1]               # set None to use cpu ; set [0] to use gpu_0
    device = torch.device('cpu')
    if gpu_id is not None:
        device = torch.device('cuda:%d' % gpu_id[0])
    checkpoint_path = '../buffer_model/IR_50_AirFace/checkpoint.tar'
    save_path = '../results/class_saliencey_vis'

    # define model
    backbone = IR_50(input_size=(112, 112), opt='E')
    head = AirFace(in_features=512, out_features=93431, device_id=gpu_id)
    backbone, head = load_pth_model(checkpoint_path, backbone, head)
    backbone = backbone.to(device)
    head = head.to(device)

    # prepare inputs
    path_ls = ['../face_imgs/3.jpg', '../face_imgs/31.jpg', '../face_imgs/8.jpg']
    labels = torch.LongTensor([34340,51979,33861])
    imgs = read_imgs(path_ls)
    labels = labels.to(device)
    imgs = imgs.to(device)

    # Class Saliency Visualization: logit-based
    sali_maps = class_saliency_vis(backbone, head, imgs, labels, device, posterior=False)
    saliency_imgs_vis(imgs, labels, sali_maps, save_path=save_path, file_name='sali_logit.jpg')
    saliency_vis_square(sali_maps, save_path=save_path, file_name='sali_logit_square.jpg')

    # Class Saliency Visualization: loss-based
    sali_maps = class_saliency_vis(backbone, head, imgs, labels, device, posterior=True)
    saliency_imgs_vis(imgs, labels, sali_maps, save_path=save_path, file_name='sali_loss.jpg')
    print('done')


if __name__ == '__main__':
    main()
