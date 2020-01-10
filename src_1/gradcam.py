"""
Created on Thu Oct 26 11:06:51 2017
@author: Utku Ozbulak - github.com/utkuozbulak

xk: refer to
Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization,
https://arxiv.org/abs/1610.02391
"""
from PIL import Image
import numpy as np
import torch
from src_1.misc_functions import get_example_params, save_class_activation_images


class CamExtractor():
    """ Extracts cam features from the model
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """ Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        for module_pos, module in self.model.features._modules.items():
            x = module(x)                               # Forward
            if int(module_pos) == self.target_layer:
                x.register_hook(self.save_gradient)     # Registers a backward hook.
                conv_output = x                         # Save the convolution output on that layer
        return conv_output, x

    def forward_pass(self, x):
        """ Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        # x = self.model.avgpool(x)   # ***** xk, according to alexnet.py, do not affect
        x = x.view(x.size(0), -1)     # Flatten
        # Forward pass on the classifier
        x = self.model.classifier(x)
        return conv_output, x


class GradCam():
    """ Produces class activation map
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Zero grads
        self.model.features.zero_grad()
        self.model.classifier.zero_grad()
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.numpy()[0]
        # Get convolution outputs
        target = conv_output.data.numpy()[0]
        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=(1, 2))        # Take averages for each gradient
        # Create empty numpy array for cam
        # ***** xk: The author first uses np.ones() for vis convenience,
        # xk has changed to use np.zeros by using 'jet' in misc_functions.save_class_activation_images()
        cam = np.zeros(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)                                # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                       input_image.shape[3]), Image.ANTIALIAS))/255
        # ^ I am extremely unhappy with this line. Originally resizing was done in cv2 which
        # supports resizing numpy matrices with antialiasing, however,
        # when I moved the repository to PIL, this option was out of the window.
        # So, in order to use resizing with ANTIALIAS feature of PIL,
        # I briefly convert matrix to PIL image and then back.
        # If there is a more beautiful way, do not hesitate to send a PR.

        # You can also use the code below instead of the code line above, suggested by @ ptschandl
        # from scipy.ndimage.interpolation import zoom
        # cam = zoom(cam, np.array(input_image[0].shape[1:])/np.array(cam.shape))
        return cam

    def generate_cam_sep(self, input_image, target_class=None):
        # written by Xiong Kai, 2019.12.30
        # Difference: Elements of feature maps use their own weight rather than the average value
        # results: the salience parts become small, more focus but less coverage (more small clusters)
        # analysis: consider only each element, lack of local correlation

        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())

        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1

        self.model.features.zero_grad()
        self.model.classifier.zero_grad()
        model_output.backward(gradient=one_hot_output, retain_graph=True)

        weights = self.extractor.gradients.data.numpy()[0]
        # weights = np.maximum(weights, 0)       # ***** xk: first filter neg grad, then average
        target = conv_output.data.numpy()[0]

        cam = weights * target
        cam = np.sum(cam, axis=0)                # ***** xk: alternatively, np.mean, np.max
        cam = np.maximum(cam, 0)                 # ***** xk: later filter, as the way used in the paper
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
        cam = np.uint8(cam * 255)
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                       input_image.shape[3]), Image.ANTIALIAS))/255

        return cam



if __name__ == '__main__':
    # Get params
    target_example = 2  # Snake
    (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
        get_example_params(target_example)
    # Grad cam
    grad_cam = GradCam(pretrained_model, target_layer=11)   # xk: layer_11 corresponds to the last nn.ReLU() in self.model.features
    # Generate cam mask
    cam = grad_cam.generate_cam(prep_img, target_class)
    cam_sep = grad_cam.generate_cam_sep(prep_img, target_class)
    # Save mask
    save_class_activation_images(original_image, cam, file_name_to_export)
    save_class_activation_images(original_image, cam_sep, file_name_to_export+'_sep')
    print('Grad cam completed')


