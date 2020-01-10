import numpy as np
import torch
from src_1.misc_functions import save_class_activation_images
# import cv2
from PIL import Image
import torchvision.transforms as transforms
from backbone.my_model_irse import IR_50
from head.airface import AirFace
from src_1.my_utils import load_pth_model


class CamExtractor():
    """ Extracts cam features from the model """
    def __init__(self, backbone, head, target_layer):
        self.model = backbone
        self.classifier = head
        self.target_layer = target_layer            # eg. target_layer='output_layer.out_dropout'
        self.register_hooks()

    def forward_hook(self,m,x,y):
        self.conv_output = y

    def backward_hook(self,m,x,y):
        self.gradients = y

    def register_hooks(self):
        eval('self.model.' + self.target_layer).register_forward_hook(self.forward_hook)
        eval('self.model.'+self.target_layer).register_backward_hook(self.backward_hook)

    def forward_pass(self, x, target_class):
        """ Does a full forward pass on the model """
        x = self.model(x)
        x = self.classifier(x, target_class)
        return x


class GradCam():
    """ Produces class activation map """
    def __init__(self, backbone, head, target_layer):
        self.model = backbone
        self.classifier = head
        self.model.eval()
        self.classifier.eval()
        self.extractor = CamExtractor(self.model, self.classifier, target_layer)

    def generate_cam(self, input_image, target_class):
        model_output = self.extractor.forward_pass(input_image, target_class)
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1

        self.model.zero_grad()
        self.classifier.zero_grad()
        model_output.backward(gradient=one_hot_output) # , retain_graph=True

        guided_gradients = self.extractor.gradients[0].data.numpy()[0]  # xk note: self.extractor.gradients is a tuple, add '[0]'
        target = self.extractor.conv_output.data.numpy()[0]
        weights = np.mean(guided_gradients, axis=(1, 2))

        cam = np.zeros(target.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)                                # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                       input_image.shape[3]), Image.ANTIALIAS))/255

        return cam

    def generate_cam_sep(self, input_image, target_class):
        # written by xk, 2019.12.30
        model_output = self.extractor.forward_pass(input_image, target_class)
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1

        self.model.zero_grad()
        self.classifier.zero_grad()
        model_output.backward(gradient=one_hot_output, retain_graph=True)

        weights = self.extractor.gradients[0].data.numpy()[0]
        target = self.extractor.conv_output.data.numpy()[0]

        cam = weights * target
        cam = np.sum(cam, axis=0)       # xk: alternatively, np.mean, np.max
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
        cam = np.uint8(cam * 255)
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                       input_image.shape[3]), Image.ANTIALIAS))/255

        return cam


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

    # Generate cam mask
    grad_cam = GradCam(backbone, head, target_layer)
    cam = grad_cam.generate_cam(input_blob, target_class)

    # Save mask
    file_name_to_export = img_path[img_path.rfind('/')+1 : img_path.rfind('.')]
    save_class_activation_images(orig_img, cam, file_name_to_export)


if __name__ == '__main__':
    main()
    print('Grad cam completed')
