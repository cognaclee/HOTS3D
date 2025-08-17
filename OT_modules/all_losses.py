import torch
from torch.autograd import Variable



### This loss is a relaxation of positive constraints on the weights
### Hence we penalize the negative ReLU

def compute_constraint_loss(list_of_params):
    loss_val = 0

    for p in list_of_params:
        loss_val += torch.relu(-p).pow(2).sum()
    return loss_val


# Computes E_y |\nabla f (\nabla g(y)) - y|^2

def inverse_constraint_loss_y_side(convex_f, convex_g, y):

    g_of_y = convex_g(y).sum()

    grad_g_of_y = torch.autograd.grad(g_of_y, y, create_graph=True)[0]

    f_grad_g_of_y = convex_f(grad_g_of_y).sum()

    grad_f_grad_g_of_y = torch.autograd.grad(f_grad_g_of_y, grad_g_of_y, create_graph=True)[0]

    constraint_loss = (grad_f_grad_g_of_y - y).pow(2).sum(dim=1).mean()

    return constraint_loss


# Assumes that x is a vector. 
# Computes E_x |\nabla g (\nabla f(x)) - x|^2

def inverse_constraint_loss_x_side(convex_f, convex_g, real_data):

    x = Variable(real_data, requires_grad=True)

    f_of_x = convex_f(x).sum()

    grad_f_of_x = torch.autograd.grad(f_of_x, x, create_graph=True)[0]

    g_grad_f_of_x = convex_g(grad_f_of_x).sum()

    grad_g_grad_f_of_x = torch.autograd.grad(g_grad_f_of_x, grad_f_of_x, create_graph=True)[0]

    constraint_loss = (grad_g_grad_f_of_x - x).pow(2).sum(dim=1).mean()

    return constraint_loss



# Assumes that both (x, y) are vectors. 
# Computes E_{(x,y)} ReLU(f(x) + g(y) + ln<x, y>)^2

def inequality_fenchel_loss(convex_f, convex_g, real_data, y):

    dims = y.shape[1]
    minus_cost = torch.log(torch.bmm(real_data.view(-1, 1, dims), y.view(-1, dims, 1)).reshape(-1, 1))

    return torch.mean((torch.clamp((convex_f(real_data) + convex_g(y) + minus_cost), min=0))**2)


# Assumes that both (x, y) are vectors
# Computes E_y |g(y) + f(\nabla g(y)) + ln<y, \nabla g(y)>|^2 + 
#          E_x |f(x) + g(\nabla f(x)) + ln<x, \nabla f(x)>|^2

def equality_fenchel_loss(grad_g_of_y, f_grad_g_y, real_data, y, convex_g, convex_f=None):

    dims = y.shape[1]
    minus_cost = torch.log(torch.bmm(grad_g_of_y.view(-1, 1, dims), y.view(-1, dims, 1)).reshape(-1, 1))

    y_transport_loss = torch.mean((f_grad_g_y + convex_g(y) + minus_cost)**2)

    # # This is for x-transport loss. This doesn't completely make sense since 'x' doesn't have a density
    if convex_f is not None:
        x = Variable(real_data, requires_grad=True)
        f_of_x = convex_f(x).sum()
        grad_f_of_x = torch.autograd.grad(f_of_x, x, create_graph=True)[0]
        n_cost = torch.log(torch.bmm(grad_f_of_x.view(-1, 1, dims), x.view(-1, dims, 1)).reshape(-1, 1))
        x_transport_loss = torch.mean((convex_g(grad_f_of_x) + convex_f(x) + n_cost)**2)
    else:
        x_transport_loss = 0

    return y_transport_loss, x_transport_loss