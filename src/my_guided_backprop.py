import torch
from torch.nn import ReLU, PReLU
from src.misc_functions import save_gradient_images, get_positive_negative_saliency
# import cv2
from PIL import Image
import torchvision.transforms as transforms
from backbone.my_model_irse import IR_50
from head.airface import AirFace
from src.my_utils import load_pth_model


class GuidedBackprop():
    """ Produces gradients generated with guided back propagation from the given image """
    def __init__(self, backbone, head):
        self.model = backbone
        self.classifier = head
        self.forward_relu_outputs = []
        self.model.eval()
        self.classifier.eval()
        self.update_relus()

    def update_relus(self):
        """ Updates relu activation functions so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero """
        def relu_backward_hook_function(module, grad_in, grad_out):
            """ If there is a negative gradient, change it to zero """
            # print(grad_in[0].size(), '  ', grad_in[1].size())  # PReLU has two inputs
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)  # xk: two guided signals
            del self.forward_relu_outputs[-1]
            return (modified_grad_out, grad_in[1])

        def relu_forward_hook_function(module, ten_in, ten_out):
            """ Store results of forward pass """
            # xk: It should be ten_in[0] to be appended according to the paper, mathematically they are
            # equal for ReLU, but not PReLU. --->  PReLU-in: ten_in[0], PReLU-out: ten_out
            self.forward_relu_outputs.append(ten_out)

        for m in self.model.modules():
            if isinstance(m, PReLU):
                m.register_backward_hook(relu_backward_hook_function)
                m.register_forward_hook(relu_forward_hook_function)

    def generate_gradients(self, input_image, target_class):
        input_image.requires_grad = True
        feat = self.model(input_image)
        model_output = self.classifier(feat, target_class)

        self.model.zero_grad()
        self.classifier.zero_grad()

        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1

        model_output.backward(gradient=one_hot_output)      # , retain_graph=True
        gradients_as_arr = input_image.grad.data.numpy()[0]
        input_image.requires_grad = False

        return gradients_as_arr


def main():
    # params setting
    img_size = (112, 112)
    target_class = torch.tensor([51979])    # label of img_crop
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

    file_name_to_export = img_path[img_path.rfind('/') + 1: img_path.rfind('.')] + '_1'

    GBP = GuidedBackprop(backbone, head)
    guided_grads = GBP.generate_gradients(input_blob, target_class)
    save_gradient_images(guided_grads, file_name_to_export + '_Guided_BP_color')

    pos_sal, neg_sal = get_positive_negative_saliency(guided_grads)
    save_gradient_images(pos_sal, file_name_to_export + '_pos_sal')
    print('Guided backprop completed')


if __name__ == '__main__':
    main()

