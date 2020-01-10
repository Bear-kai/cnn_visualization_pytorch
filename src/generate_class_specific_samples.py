"""
Created on Thu Oct 26 14:19:44 2017
@author: Utku Ozbulak - github.com/utkuozbulak

xk: refer to
Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps, ICLR2014
https://arxiv.org/abs/1312.6034
related to my_class_model_vis.py
"""
import os
import numpy as np
from torch.optim import SGD
from torchvision import models
from src.misc_functions import preprocess_image, recreate_image, save_image


class ClassSpecificImageGeneration():
    """
        Produces an image that maximizes a certain class with gradient ascent
    """
    def __init__(self, model, target_class):
        self.model = model
        self.model.eval()
        self.target_class = target_class

        # Generate a random image --> xk: start from zeros may be better
        # self.created_image = np.uint8(np.random.uniform(0, 255, (224, 224, 3)))
        self.created_image = np.zeros((224,224,3),dtype=np.uint8)

        # Create the folder to export images if not exists
        if not os.path.exists('../results/generated'):
            os.makedirs('../results/generated')

    def generate(self):
        initial_learning_rate = 6
        for i in range(1, 101):
            # Process image and return variable
            self.processed_image = preprocess_image(self.created_image, False)
            # Define optimizer for the image
            optimizer = SGD([self.processed_image], lr=initial_learning_rate)
            # Forward
            output = self.model(self.processed_image)   # xk: torch.Size([1,1000]), logit-based, no regularization
            # Target specific class
            class_loss = -output[0, self.target_class]  # xk: max_logit --> min_-logit
            print('Iteration:', str(i), 'Loss', "{0:.2f}".format(class_loss.data.numpy()))
            # Zero grads
            self.model.zero_grad()
            # Backward
            class_loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = recreate_image(self.processed_image)
            if i % 10 == 0:
                # Save image
                im_path = '../results/generated/c_specific_iteration_'+str(i)+'.jpg'
                save_image(self.created_image, im_path)
        return self.processed_image


if __name__ == '__main__':
    target_class = 130  # Flamingo
    pretrained_model = models.alexnet(pretrained=True)
    csig = ClassSpecificImageGeneration(pretrained_model, target_class)
    csig.generate()
