import torch
import torch.nn.functional as F


def NeighborhoodD(dcos, idx):
    b, n = idx.shape[0], idx.shape[1]
    k = idx.shape[-1]
    dcos = dcos.view(-1, n).unsqueeze(-1)
    idx = idx.view(-1, k).unsqueeze(1)
    dcollected = gather_features(dcos, idx).view(b, n, k)
    return dcollected


def gather_features(x, indices):
    # x:(N, F) indices:(N, k) -> (N, k, F)
    x = x.unsqueeze(-2).expand(*x.shape[:-1], indices.shape[-1], x.shape[-1])
    indices = indices.unsqueeze(-1).expand(*indices.shape, x.shape[-1])
    return x.gather(-3, indices)


def NeighborhoodGraphMAD(dcos, idx):
    D = NeighborhoodD(dcos, idx)
    return nanmean(D, dim=-1)


def nanmean(v, *args, inplace=False, **kwargs):
    if not inplace:
        v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)


def all_distances(A, B):
    r_A = torch.sum(A * A, dim=2, keepdim=True)
    r_B = torch.sum(B * B, dim=2, keepdim=True)
    m = torch.matmul(A, B.permute(0, 2, 1))
    D = r_A - 2 * m + r_B.permute(0, 2, 1)
    return D


def pairwise_cosine_similarity(a, b):
    a_norm, b_norm = F.normalize(a, dim=-1), F.normalize(b, dim=-1)
    d = torch.matmul(a_norm, b_norm.transpose(-1, -2))
    return d


def dilated_stochastic_nn(d, k=16, dilation=1, epsilon=0.2, largest=False, stochastic=False):
    nn_idx = d.topk(k * dilation, largest=largest)[1]
    if stochastic:
        if torch.rand(1) < epsilon:
            num = k * dilation
            randnum = torch.randperm(num)[:k]
            nn_idx = nn_idx[:, :, randnum]
        else:
            nn_idx = nn_idx[:, :, ::dilation]
    else:
        nn_idx = nn_idx[:, :, ::dilation]

    return nn_idx


def batchwise_MADGap(fts, k, dilation=1):
    d = all_distances(fts, fts)
    idxFarthest = dilated_stochastic_nn(d, k, dilation, largest=True)
    idxClosest = dilated_stochastic_nn(d, k, dilation, largest=False)

    dcos = 1 - pairwise_cosine_similarity(fts, fts)

    mean_MAD = torch.mean(dcos)
    MADClosest = nanmean(torch.gather(dcos, dim=2, index=idxClosest), dim=-1, inplace=True)
    MADFarthest = nanmean(torch.gather(dcos, dim=2, index=idxFarthest), dim=-1, inplace=True)

    return mean_MAD, MADFarthest, MADClosest

