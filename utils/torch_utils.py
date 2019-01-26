import torch


def check_cuda():
    return torch.cuda.is_available()


CUDA_AVAILABLE = check_cuda()


def select_device(force_cpu=False):
    if force_cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0' if CUDA_AVAILABLE else 'cpu')
    return device
