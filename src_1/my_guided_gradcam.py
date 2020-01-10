import torch
import numpy as np
from src_1.misc_functions import save_gradient_images
from src_1.my_gradcam import GradCam
from src_1.my_guided_backprop import GuidedBackprop
from src_1.my_utils import load_pth_model
# import cv2
from PIL import Image
import torchvision.transforms as transforms
from backbone.my_model_irse import IR_50
from head.airface import AirFace


def guided_grad_cam(grad_cam_mask, guided_backprop_mask):
    cam_gb = np.multiply(grad_cam_mask, guided_backprop_mask)
    return cam_gb


def main():
    # params setting
    img_size = (112, 112)
    target_class = torch.tensor([51979])    # label of img_crop
    target_layer = 'output_layer.out_dropout'
    img_path = '../face_imgs/img_crop.jpg'  # img.png
    checkpoint_path = '../buffer_model/IR_50_AirFace/checkpoint.tar'

    # read and process sample image
    val = [0.5,0.5,0.5]
    preprocess = transforms.Compose([# transforms.ToPILImage(),             # if use cv2.imread
                                     # transforms.Resize(size=img_size),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=val, std=val)])
    input_blob = torch.zeros((1, 3, 112, 112))         # , requires_grad=True
    # img = cv2.imread('cropped_img_1.jpg')            # h*w*c
    # img = img[:, :, ::-1].copy()                     # to rgb, or img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_img = Image.open(img_path).convert('RGB')    # PIL: jpg->RGB,png->RGBA, CV2: BGR, PIL and CV2 have minor value differences
    orig_img = orig_img.resize(img_size)               # orig_img.save('./test_face_img.jpg')  # will be compressed
    img = preprocess(orig_img)                         # c*h*w
    input_blob[0] = img

    # define model
    backbone = IR_50(input_size=img_size, opt='E')
    head = AirFace(in_features=512, out_features=93431, device_id=None)
    backbone, head = load_pth_model(checkpoint_path, backbone, head)

    grad_cam = GradCam(backbone, head, target_layer)
    cam = grad_cam.generate_cam(input_blob, target_class)
    print('Grad cam completed')

    GBP = GuidedBackprop(backbone, head)
    guided_grads = GBP.generate_gradients(input_blob, target_class)
    print('Guided backpropagation completed')

    file_name_to_export = img_path[img_path.rfind('/') + 1: img_path.rfind('.')] # + '_1'
    cam_gb = guided_grad_cam(cam, guided_grads)
    save_gradient_images(cam_gb, file_name_to_export + '_myGuidedGradCam')
    print('Guided grad cam completed')


if __name__ == '__main__':
    main()
