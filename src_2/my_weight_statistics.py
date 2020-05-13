import torch
import torch.nn as nn
from backbone.my_model_irse import IR_50
from src_1.my_utils import load_pth_model
import matplotlib.pyplot as plt
import os


def collect_weight_norm(model):
    weight_dic = {'conv_weight_norm':[], 'conv_weight_size':[],
                  'bn_weight_norm':[],'bn_weight_size':[],
                  'fc_weight_norm':[],'fc_weight_size':[]
                  }
    bias_dic = {'conv_bias_norm':[], 'conv_bias_size':[],
                'bn_bias_norm':[], 'bn_bias_size':[],
                'fc_bias_norm':[], 'fc_bias_size':[]
                }
    running_dic = {'mean':[], 'var':[]}
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            weight_dic['conv_weight_norm'].append(torch.norm(m.weight.data))
            weight_dic['conv_weight_size'].append(tuple(m.weight.shape))
            if m.bias is not None:
                bias_dic['conv_bias_norm'].append(torch.norm(m.bias.data))
                bias_dic['conv_bias_size'].append(tuple(m.weight.shape))
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            weight_dic['bn_weight_norm'].append(torch.norm(m.weight.data))
            weight_dic['bn_weight_size'].append(tuple(m.weight.shape))
            bias_dic['bn_bias_norm'].append(torch.norm(m.bias.data))
            bias_dic['bn_bias_size'].append(tuple(m.weight.shape))
            running_dic['mean'].append(torch.mean(m._buffers['running_mean']))
            running_dic['var'].append(torch.mean(m._buffers['running_var']))
        elif isinstance(m, nn.Linear):
            weight_dic['fc_weight_norm'].append(torch.norm(m.weight.data))
            weight_dic['fc_weight_size'].append(tuple(m.weight.shape))
            if m.bias is not None:
                bias_dic['fc_bias_norm'].append(torch.norm(m.bias.data))
                bias_dic['fc_bias_size'].append(tuple(m.weight.shape))
    return weight_dic, bias_dic, running_dic



def draw_and_save(weight_dic, bias_dic, running_dic, save_path, file_name):

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    fig, axs = plt.subplots(2, 4, figsize=(12,6))
    axs[0,0].plot(weight_dic['conv_weight_norm'])
    axs[0,0].grid()
    axs[0,0].set_title('conv_weight_norm')

    axs[0,1].plot(weight_dic['bn_weight_norm'])
    axs[0,1].grid()
    axs[0,1].set_title('bn_weight_norm')

    axs[0,2].plot(weight_dic['fc_weight_norm'])
    axs[0,2].grid()
    axs[0,2].set_title('fc_weight_norm')

    axs[0,3].plot(running_dic['mean'])
    axs[0,3].grid()
    axs[0,3].set_title('running_mean')

    axs[1,0].plot(bias_dic['conv_bias_norm'])
    axs[1,0].grid()
    axs[1,0].set_title('conv_bias_norm')

    axs[1,1].plot(bias_dic['bn_bias_norm'])
    axs[1,1].grid()
    axs[1,1].set_title('bn_bias_norm')  # , fontsize=10

    axs[1,2].plot(bias_dic['fc_bias_norm'])
    axs[1,2].grid()
    axs[1,2].set_title('fc_bias_norm')

    axs[1,3].plot(running_dic['var'])
    axs[1,3].grid()
    axs[1,3].set_title('running_var')

    # fig.suptitle('%s' % name)  # , fontsize=10, y=1.08    overlap...
    plt.tight_layout()
    plt.savefig(os.path.join(save_path,'%s.jpg'%file_name))

    weight_keys = ['conv_weight_size', 'bn_weight_size', 'fc_weight_size']
    bias_keys = ['conv_bias_size', 'bn_bias_size', 'fc_bias_size']
    with open(os.path.join(save_path,'%s.txt'%file_name), 'a') as file_txt:
        file_txt.write('\n\n')
        for i,key in enumerate(weight_keys):
            if len(bias_dic[bias_keys[i]]):
                file_txt.write('===== %s | %s =====' % (key, bias_keys[i]) + '\n')
                for j,term in enumerate(weight_dic[key]):
                    file_txt.write(str(term) + '\t' + str(bias_dic[bias_keys[i]][j]) + '\n')
            else:
                file_txt.write('===== %s | have no %s =====' % (key, bias_keys[i]) + '\n')
                for term in weight_dic[key]:
                    file_txt.write(str(term) + '\n')

    return 0


def main_1():

    checkpoint_path = '../buffer_model/IR_50_AirFace/checkpoint.tar'
    backbone = IR_50(input_size=(112, 112), opt='E')
    backbone, _ = load_pth_model(checkpoint_path, backbone)

    save_path = '../results/src_2/weight_norm_statistics'
    file_name = 'IR50_norm_statistics'
    weight_dic, bias_dic, running_dic = collect_weight_norm(backbone)
    draw_and_save(weight_dic, bias_dic, running_dic, save_path, file_name)
    print('done')



def main_2():

    from torchvision import models
    alex_model = models.alexnet(pretrained=True)

    save_path = '../results/src_2/weight_norm_statistics'
    file_name =  'Alex_norm_statistics'
    weight_dic, bias_dic, running_dic = collect_weight_norm(alex_model)
    draw_and_save(weight_dic, bias_dic, running_dic, save_path, file_name)
    print('done')


if __name__ == '__main__':
    # main_1()
    main_2()
