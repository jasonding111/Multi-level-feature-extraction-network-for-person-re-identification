import torch
from utils.ranger import Ranger


def build_optimizer(model, center_criterion):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = 0.006
        weight_decay = 0.0005
        if "gap" in key:
            lr = lr * 10
            weight_decay = 0
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    optimizer = Ranger(params)
    print('using Ranger for optimizer ')
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=0.5)

    return optimizer, optimizer_center
