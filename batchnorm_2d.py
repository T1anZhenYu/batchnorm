import torch 
import torch.nn as nn
class MyBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.9, affine=True):
        """
        Input Variables:
        ----------------
            beta, gamma, tau: Variables of shape [1, C, 1, 1].
            eps: A scalar constant or learnable variable.
        """

        super(MyBatchNorm2d, self).__init__()
        self.affine = affine
        if self.affine:
            self.beta = nn.parameter.Parameter(
                torch.Tensor(1, num_features, 1, 1), requires_grad=True)
            self.gamma = nn.parameter.Parameter(
                torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        else:
            self.beta = nn.parameter.Parameter(
                torch.Tensor(1, num_features, 1, 1), requires_grad=False)
            self.gamma = nn.parameter.Parameter(
                torch.Tensor(1, num_features, 1, 1), requires_grad=False)            
        self.eps = torch.Tensor(eps)
        self.running_mean = torch.Tensor(1, num_features, 1, 1)
        self.running_var = torch.Tensor(1, num_features, 1, 1)
        self.mbn2 = mbn2.apply
        self.momentum = momentum
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)
        nn.init.ones_(self.running_var)
        nn.init.zeros_(self.running_var)
    def forward(self, x):
        if self.training:
            x, mean, var = self.mbn2(x,self.beta,self.gamma,self.affine,self.eps)
            self.running_var = self.momentum * self.running_var + (1 - self.momentum)*var
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum)*mean
        else:
            x = self.gamma * (x - self.running_mean)/(torch.sqrt(self.running_var)) + self.beta
        return x





class mbn2(torch.autograd.Function):
    @staticmethod
    def forward(self, x, beta, gamma,affine, eps):

        mean = x.mean(dim=(0,2,3), keepdim=True)
        var = (x - mean).pow(2).mean(dim=(0,2,3),keepdim=True)
        x_hat = gamma * (x - mean) / torch.sqrt(var + eps) + beta
        self.save_for_backward(x, beta, gamma, mean, var, x_hat, affine, eps)
        return x_hat, mean, var

    @staticmethod
    def backward(self, grad_output):
        x, beta, gamma, mean, var, x_hat, affine, eps= self.saved_tensors
        N = x.shape[0]
        if affine:
            dgamma = torch.sum(grad_output * x_hat,dim=0, keepdim=True)
            dbeta = torch.sum(grad_output,dim=0,keepdim=True)
        else:
            dgamma = None
            dbeta = None
        dx_hat = dout * gamma
        dsigma = -0.5 * torch.sum(dx_hat * (x - mean),dim=0,keepdim=True) * (var + eps).pow(-1.5)
        dmu = torch.sum(dx_hat / torch.sqrt(var + eps),dim=0,keepdim=True) - 2 * dsigma * torch.sum(x - mean,dim=0,keepdim=True) / N
        dx = dx_hat / torch.sqrt(var + eps) + 2 * dsigma * (x - mean) / N + dmu / N 

        return dx, dbeta, dgamma, None, None, None, None, None