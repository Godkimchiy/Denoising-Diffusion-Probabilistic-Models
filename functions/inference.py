import torch


# def compute_alpha(beta, t):
#     beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
#     a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
#     return a


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))