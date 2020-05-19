import torch


def adj_to_bias(adj, nhood=1):
    mt = torch.eye(adj.shape[0]).to(DEVICE)
    for _ in range(nhood):
        mt = torch.matmul(mt, (adj + torch.eye(adj.shape[0]).to(DEVICE)))
    mt = torch.clamp(mt, max=1.0)

    return mt