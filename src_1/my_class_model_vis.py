import torch
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from backbone.my_model_irse import IR_50
from head.airface import AirFace
from src_1.my_utils import load_pth_model, postprocess


def class_model_vis(backbone,head, y, device,lam=5,p=2,epoch=450,lr=0.1,posterior=False,save_path=''):
    """ function that generates img of a specified class by gradient descent
        default to save 9 generated imgs and 1 curve img during the training process
        input:
                backbone: pre-trained network body, output feature vector
                head: pre-trained network head, output class logits
                y: torch.tensor, img of which class to generate
                device: specify cpu or gpu
                lam: the regularization param
                p: p-norm param
                epoch: optimization steps, should be multiple of 9
                lr: learning rate
                posterior: True--loss based optimization, False--logit based optimization
                save_path: path to save generated img and optimiaztion curves
        return:
                x: generated img: torch.tensor
    """
    if epoch % 9:
        raise ValueError('Set epoch=9*n where n is 1,2,3,4...')
    if not save_path:
        raise ValueError('Please set save_path')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    flag_bk_save = backbone.training
    flag_hd_save = head.training
    backbone.eval()
    head.eval()

    div_num = epoch//9
    stages = [div_num*3, div_num*6]
    fig, axs = plt.subplots(3, 3, figsize=(9, 9))

    x = torch.zeros(1,3,112,112).to(device)     # generate one img of a specified class
    x.requires_grad = True

    alter_name = 'logit'
    if posterior:
        loss_func = torch.nn.CrossEntropyLoss()
        alter_name = 'loss'

    optimizer_1 = optim.SGD([{'params': backbone.parameters()}, {'params': head.parameters()}], lr=1e-6)
    optimizer_2 = optim.SGD([x],lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_2, milestones=[*stages], gamma=0.1)

    val_ls = []             # list of loss/score
    val_ls_total = []       # list of loss/socre with regularization norm
    xnm_ls = []             # list of regularization norm
    for i in range(epoch):
        feat = backbone(x)
        logits = head(feat, y)
        if posterior:
            loss = loss_func(logits, y)
            xnm = x.view(1,-1).norm(p=p,dim=1).squeeze()
            alter_val = loss.item()
            loss += lam * xnm
            optimizer_1.zero_grad()
            optimizer_2.zero_grad()
            # to generate imgs of multi-classes at the same time, try to use:
            # loss.backward(torch.ones(loss.shape).to(device))
            loss.backward()
            # x.data -= lr * x.grad.data
            optimizer_2.step()
            lr_scheduler.step()
            alter_val_total = loss.item()
        else:
            score = logits.gather(1, y.view(-1, 1)).squeeze()
            xnm = x.view(1,-1).norm(p=p,dim=1).squeeze()
            alter_val = score.item()
            score -= lam * xnm
            score = -1.0 * score
            optimizer_1.zero_grad()
            optimizer_2.zero_grad()
            score.backward()
            # x.data -= lr * x.grad.data
            optimizer_2.step()
            lr_scheduler.step()
            alter_val_total = score.item()

        if (i+1) % div_num == 0:
            ind = (i+1)//div_num - 1
            # since x requires_grad=True, need .data() or .detach() before .numpy()
            axs[ind//3, ind%3].imshow(postprocess(x[0].data.cpu().numpy().transpose(1,2,0)))
            axs[ind//3, ind%3].axis('off')
            axs[ind//3, ind%3].set_title('iter=%d, %s=%.3f'%(i+1,alter_name,alter_val))
            if (i+1)//div_num==9:
                plt.tight_layout()
                plt.savefig(os.path.join(save_path, 'gen_imgs_%s.jpg'%alter_name))
                plt.close()

        val_ls.append(alter_val)
        val_ls_total.append(alter_val_total)
        xnm_ls.append(xnm)
        print('iter %d/%d, val_total %.3f'%(i+1,epoch,alter_val_total))

    l1, = plt.plot(val_ls)
    l2, = plt.plot(val_ls_total)
    l3, = plt.plot(xnm_ls)
    plt.legend(handles=[l1,l2,l3,], labels=['%s'%alter_name, 'Total', 'norm'], loc='best')
    plt.savefig(os.path.join(save_path, 'curves_%s.jpg'%alter_name))
    plt.close()

    backbone.zero_grad()
    head.zero_grad()
    backbone.train(flag_bk_save)
    head.train(flag_hd_save)
    x.grad.detach_()
    x.grad = None
    x.requires_grad = False

    return x.cpu()  # .numpy()


def main():
    # param setting
    gpu_id = [1]               # set None to use cpu ; set [0] to use gpu_0
    device = torch.device('cpu')
    if gpu_id is not None:
        device = torch.device('cuda:%d' % gpu_id[0])
    checkpoint_path = '../buffer_model/IR_50_AirFace/checkpoint.tar'
    save_path = '../results/class_model_vis'

    # define model
    backbone = IR_50(input_size=(112, 112), opt='E')
    head = AirFace(in_features=512, out_features=93431, device_id=gpu_id)
    backbone, head = load_pth_model(checkpoint_path, backbone, head)
    backbone = backbone.to(device)
    head = head.to(device)

    # img of which class to generate     # 31:51979  3：34340  8:33861
    y = torch.LongTensor([51979]).to(device)

    # Class Model Visualization: logit-based
    class_model_vis(backbone, head, y, device, posterior=False,save_path=save_path)
    # Class Model Visualization: loss-based
    class_model_vis(backbone, head, y, device, posterior=True,save_path=save_path)
    print('done')


if __name__ == '__main__':
    torch.backends.cudnn.deterministic = True   # ensure to have reproducible results
    main()
