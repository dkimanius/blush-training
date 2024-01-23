import torch
import numpy as np
from contextlib import nullcontext

FORWARD_ITERS = np.array([1, 2, 3, 4, 5])
FORWARD_ITERS_P = np.array([10, 6, 3, 2, 1], dtype=float)
FORWARD_ITERS_P /= np.sum(FORWARD_ITERS_P)


def loss_function(model, sample):
    input1 = sample["input1"]
    input2 = sample["input2"]
    output1 = sample["output1"]
    output2 = sample["output2"]
    rmask = sample["rmask"]

    num_iter = np.random.choice(FORWARD_ITERS, p=FORWARD_ITERS_P)

    infer1 = input1
    for i in range(num_iter):
        with torch.no_grad() if i != num_iter - 1 else nullcontext():
            infer1 = model(infer1)
    lp_loss1 = rmask * torch.nn.functional.mse_loss(output2, infer1, reduction='none')
    b_loss = torch.mean(lp_loss1, (1, 2, 3))

    infer2 = input2
    for i in range(num_iter):
        with torch.no_grad() if i != num_iter - 1 else nullcontext():
            infer2 = model(infer2)
    lp_loss2 = rmask * torch.nn.functional.mse_loss(output1, infer2, reduction='none')
    b_loss += torch.mean(lp_loss2, (1, 2, 3))

    loss = b_loss.mean()

    return loss
