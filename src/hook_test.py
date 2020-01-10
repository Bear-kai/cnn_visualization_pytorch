import torch
import torch.nn as nn
# tested by xk, 2019.12.31
# https://github.com/pytorch/pytorch/issues/598


def back_hook(m,go,gi):
    print("Grad Input")     # grad of Loss w.r.t the forward input of M
    print(go)
    print("Grad Output")    # grad of Loss w.r.t the forward output of M
    print(gi)


class M(nn.Module):
    def __init__(self):
        super(M,self).__init__()
        self.register_backward_hook(back_hook)

    def forward(self,x,y,z):
        return (2*x+y+2*z)  # p


x=torch.randn(1,2,requires_grad=True)   # data: tensor([[-0.0354,  1.8845]])   grad: tensor([[-3.2813,  3.2648]])
y=torch.randn(1,2,requires_grad=True)   # data: tensor([[-1.4789, -0.4045]])   grad: tensor([[-1.6406,  1.6324]])
z=torch.randn(1,2,requires_grad=True)   # data: tensor([[-0.5071, -1.3419]])   grad: tensor([[-3.2813,  3.2648]])

criterion=nn.MSELoss()
mod=M()
out=mod(x,y,z)
label = torch.randn(1,2)    # q = tensor([[-0.9233, -0.9517]])
loss=criterion(out,label)
loss.backward()
print('done')

# ===== Results:
# Grad Input:  (tensor([[-1.6406,  1.6324]]), tensor([[-1.6406,  1.6324]]))
# Grad Output: (tensor([[-1.6406,  1.6324]]),)
# ===== Notes:
# actually, module hooks are actually registered on the last function that the module has created.
# In above case, the hook is registered on that ((2*x+y)+(2*z)) --> (A)+(B) add operation
# This is why it gets only two grad inputs, and grad_in equals to grad_out.
# Note that the grads of x,y,and z are computed correctly, say 2*grad_output, grad_output, 2*grad_output
# ===== Detail Analysis:
# p = 2*x+y+2*z = [-2.5639,0.6807]; L = mean_(p-q)^2 --> L = 1/2*[(p1-q1)^2 + (p2-q2)^2]
# partial_L/partial_p1 = p1-q1 = -2.5639 + 0.9233 = -1.6406
# partial_L/partial_p2 = p2-q2 = 0.6807 + 0.9517 = 1.6324     ---> grad output [-1.6406,  1.6324]
# p = (2*x+y) + (2*z) = A + B
# partial_L/partial_A1 = partial_L/partial_p1 * partial_p1/partial_A1 = -1.6406 * 1
# partial_L/partial_A2 = ...                                  ---> grad input tensor 1
# partial_L/partial_B1 = ...
# partial_L/partial_B2 = ...                                  ---> grad input tensor 2

# other ref:
# https://oldpan.me/archives/pytorch-autograd-hook
# https://www.cnblogs.com/hellcat/p/8512090.html
# https://discuss.pytorch.org/t/exact-meaning-of-grad-input-and-grad-output/14186/3

# warning from pytorch doc:
#     The current implementation will not have the presented behavior
#     for complex :class:`Module` that perform many operations.
#     In some failure cases, :attr:`grad_input` and :attr:`grad_output` will only
#     contain the gradients for a subset of the inputs and outputs.
#     For such :class:`Module`, you should use :func:`torch.Tensor.register_hook`
#     directly on a specific input or output to get the required gradients.
