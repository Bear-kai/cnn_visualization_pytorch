import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2


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


def read_imgs(path_ls,size=(112,112)):
    """ input path_ls: list of img path """
    preprocess = transforms.Compose([transforms.ToPILImage(),
                                     transforms.Resize(size=size),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    imgs = []
    for img_path in path_ls:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = preprocess(img).unsqueeze(0)  # [1,3,112,112]
        imgs.append(img)
    imgs = torch.cat(imgs,dim=0)
    return imgs


def postprocess(images):
    """ input imgs: numpy.ndarray, format: (N*)H*W*C"""
    imgs = images.copy()                        # Pytorch: .clone()
    if len(imgs.shape) == 3:
        imgs = np.expand_dims(imgs, axis=0)     # Pytorch: imgs = imgs.unsqueeze(0)
    N = imgs.shape[0]
    img_arr = np.zeros(imgs.shape,dtype=np.uint8)
    for i in range(N):
        img = imgs[i]
        img = img * 0.5 + 0.5
        img = (img - img.min()) / (img.max() - img.min()) * 255
        # modify img_arr will not affect img
        img_arr[i] = img.astype(np.uint8)
    img_arr = img_arr.squeeze()

    return img_arr


def saliency_imgs_vis(images, labels, sali_maps, save_path='',file_name='sample.jpg'):
    """ Save figure of original image and its corresponding saliency map.
        Better looking results than saliency_vis_square().
        input:
            images, labels: torch.tensor
            sali_maps: numpy.ndarray
    """
    if not save_path:
        raise ValueError('Please set save_path')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # after '.cpu()', modify imgs will not affect images
    imgs = images.cpu().numpy()
    N = imgs.shape[0]
    if N==1:
        fig, axs = plt.subplots(2, N, figsize=(3, 6))
        axs[0].imshow(postprocess(imgs[0].transpose(1,2,0)))
        axs[0].axis('off')
        axs[0].set_title('label-%d'%labels[0])
        # plt.cm refer to: https://matplotlib.org/examples/color/colormaps_reference.html
        # Three ways to set cmap: cmap=plt.get_cmap('gray_r'), cmap='gray_r', cmap=plt.cm.binary
        axs[1].imshow(sali_maps[0], cmap=plt.cm.hot)
        axs[1].axis('off')
    elif N>1:
        fig, axs = plt.subplots(2, N, figsize=(N*3, 6))
        for i in range(N):
            # a=b.transpose(), modify a will affect b
            axs[0,i].imshow(postprocess(imgs[i].transpose(1,2,0)))
            axs[0,i].axis('off')
            axs[0,i].set_title('label-%d' % labels[i])
            axs[1, i].imshow(sali_maps[i], cmap=plt.cm.hot)
            axs[1,i].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path,file_name))
    plt.close()


def saliency_vis_square(data, padsize=5, padval=0, save_path='',file_name='sample.jpg'):
    """ Save several saliency maps into one square figure.
        Note: The function may produce different looking results compared to function saliency_imgs_vis(),
              since the saliency maps affect each other when they are showed in one figure with hot style.
        input
            data: torch.tensor or numpy.ndarray. support data of N*H*W and N*H*W*C
    """
    if not save_path:
        raise ValueError('Please set save_path')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    plt.figure(figsize=(9,9))
    plt.imshow(data, cmap=plt.cm.hot)  # gray
    plt.axis('off')
    plt.savefig(os.path.join(save_path,file_name))
    plt.close()

