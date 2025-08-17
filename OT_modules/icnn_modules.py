import torch.nn as nn
from torch.autograd import Function
import torch


def get_activation(activation):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'elu':
        return nn.ELU()
    elif activation == 'leaky_relu':
        return nn.LeakyReLU(0.2)
    elif activation == 'celu':
        return nn.CELU()
    elif activation == 'selu':
        return nn.SELU()
    elif activation == 'tanh':
        return nn.Tanh()
    else:
        raise NotImplementedError('activation [%s] is not found' % activation)


class ConvexLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        # self.wb = kargs[0]
        # self.have_alpha = kargs[1]
        #
        # kargs = kargs[2:]

        super(ConvexLinear, self).__init__(*kargs, **kwargs)

        if not hasattr(self.weight, 'be_positive'):
            self.weight.be_positive = 1.0

    def forward(self, input):

        out = nn.functional.linear(input, self.weight, self.bias)

        return out

class LORALinear(nn.Module):

    def __init__(self, input_dim, output_dim, bias=True, rank=128):

        super(LORALinear, self).__init__()
        self.r = rank
        self.lora_A = nn.Parameter(torch.randn((input_dim, self.r)))
        self.lora_B = nn.Parameter(torch.randn((self.r, output_dim)))
        if bias:
            self.bias = nn.Parameter(torch.randn(output_dim))
        else:
            self.bias=None


    def forward(self, input):
        print('input.shape',input.shape)
        print('self.lora_A.shape',self.lora_A.shape)
        print('self.lora_B.shape',self.lora_B.shape)
        LORA = self.lora_A@self.lora_B
        print('self.lora_C.shape',LORA.shape)
        out = nn.functional.linear(input, LORA.T,self.bias)

        return out


class ConvexLORALinear(nn.Module):

    def __init__(self, input_dim, output_dim, rank=128):

        super(ConvexLORALinear, self).__init__()
        self.r = rank
        self.lora_A = nn.Parameter(torch.pow(torch.randn((input_dim, self.r)),2))
        self.lora_B = nn.Parameter(torch.pow(torch.randn((self.r, output_dim)),2))

        if not hasattr(self.lora_A, 'be_positive'):
            self.lora_A.be_positive = 1.0
            self.lora_B.be_positive = 1.0
            

    def forward(self, input):

        out = nn.functional.linear(input, (self.lora_A@self.lora_B).transpose(0, 1))

        return out

class ConvexConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):

        super(ConvexConv2d, self).__init__(*kargs, **kwargs)

        if not hasattr(self.weight, 'be_positive'):
            self.weight.be_positive = 1.0

    def forward(self, input):

        out = nn.functional.conv2d(input, self.weight, self.bias, self.stride,
                                   self.padding, self.dilation, self.groups)
        return out


class ICNN_Quadratic(nn.Module):

    def __init__(self, layers_n, input_dim, hidden_dim, activation,full_quadratic=False,rank=64):

        super(ICNN_Quadratic, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.layers_n = layers_n
        self.full_quad = full_quadratic
        self.rank = rank

        # x -> h_1  
        
        self.fc_normals = [nn.Linear(self.input_dim, self.hidden_dim, bias=True)]
        self.fc_convexs = []
        self.activs = [get_activation(self.activation)]
        for i in range(1,layers_n):
            self.fc_normals.append(nn.Linear(self.input_dim, self.hidden_dim, bias=True))
            self.activs.append(get_activation(self.activation))
            self.fc_convexs.append(ConvexLinear(self.hidden_dim, self.hidden_dim, bias=False))
            #self.fc_convexs.append(ConvexLORALinear(self.hidden_dim, self.hidden_dim))

        self.fc_convexs.append(ConvexLinear(self.hidden_dim, 1, bias=False))
        self.fc_normals.append(nn.Linear(self.input_dim, 1, bias=True))
        '''
        self.fc_normals = [LORALinear(self.input_dim, self.hidden_dim, bias=True, rank=self.rank)]
        self.fc_convexs = []
        self.activs = [get_activation(self.activation)]
        for i in range(1,layers_n):
            self.fc_normals.append(LORALinear(self.input_dim, self.hidden_dim, bias=True, rank=self.rank))
            self.activs.append(get_activation(self.activation))
            self.fc_convexs.append(ConvexLORALinear(self.hidden_dim, self.hidden_dim, rank=self.rank))
            self.hidden_dim = self.hidden_dim//2

        self.fc_convexs.append(ConvexLORALinear(self.hidden_dim, 1, rank=self.rank))
        self.fc_normals.append(LORALinear(self.input_dim, 1, bias=True, rank=self.rank))
        '''
        self.fc_normals = nn.ModuleList(self.fc_normals)
        self.fc_convexs = nn.ModuleList(self.fc_convexs)
        self.activs = nn.ModuleList(self.activs) 
        



    # Input is of size
    def forward(self, input):

        x = self.activs[0](self.fc_normals[0](input)).pow(2)
        for i in range(1,self.layers_n):
            x = self.activs[i](self.fc_convexs[i-1](x).add(self.fc_normals[i](input)))

        if self.full_quad:
            x = self.fc_convexs[-1](x).add(self.fc_normals[-1](input)).pow(2)
        else:
            x = self.fc_convexs[-1](x).add(self.fc_normals[-1](input).pow(2))

        return x

