import numpy as np
import torch

from torch_geometric.data import Data
import torch_points_kernels as tp

from .torch_nn import MLP


class DenseFPModule(torch.nn.Module):
    def __init__(self, up_conv_nn, norm="batch", act="leakyrelu", **kwargs):
        super(DenseFPModule, self).__init__()

        self.nn = MLP(up_conv_nn, norm=norm, act=act, bias=False)

    def conv(self, pos, pos_skip, x):
        assert pos_skip.shape[2] == 3

        if pos is not None:
            dist, idx = tp.three_nn(pos_skip, pos)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_feats = tp.three_interpolate(x, idx, weight)
        else:
            interpolated_feats = x.expand(*(x.size()[0:2] + (pos_skip.size(1),)))

        return interpolated_feats

    def forward(self, data, **kwargs):
        """ Propagates features from one layer to the next.
        data contains information from the down convs in data_skip

        Arguments:
            data -- (data, data_skip)
        """
        data, data_skip = data
        pos, x = data.pos, data.x.squeeze(-1)
        pos_skip, x_skip = data_skip.pos, data_skip.x.squeeze(-1)

        new_features = self.conv(pos, pos_skip, x)

        if x_skip is not None:
            new_features = torch.cat([new_features, x_skip], dim=1)  # (B, C2 + C1, n)

        if hasattr(self, "nn"):
            new_features = self.nn(new_features)

        return Data(x=new_features.squeeze(-1), pos=pos_skip)


class GlobalDenseBaseModule(torch.nn.Module):
    def __init__(self, nn, aggr="max", norm="batch", act="leakyrelu", **kwargs):
        super(GlobalDenseBaseModule, self).__init__()
        self.nn = MLP(nn, norm=norm, act=act, bias=False)
        if aggr.lower() not in ["mean", "max"]:
            raise Exception("The aggregation provided is unrecognized {}".format(aggr))
        self._aggr = aggr.lower()

    @property
    def nb_params(self):
        """[This property is used to return the number of trainable parameters for a given layer]
        It is useful for debugging and reproducibility.
        Returns:
            [type] -- [description]
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        self._nb_params = sum([np.prod(p.size()) for p in model_parameters])
        return self._nb_params

    def forward(self, data, **kwargs):
        x, pos = data.x.squeeze(-1), data.pos
        pos_flipped = pos.transpose(1, 2).contiguous()

        x = self.nn(torch.cat([x, pos_flipped], dim=1).unsqueeze(-1))

        if self._aggr == "max":
            x = x.squeeze(-1).max(-1)[0]
        elif self._aggr == "mean":
            x = x.squeeze(-1).mean(-1)
        else:
            raise NotImplementedError("The following aggregation {} is not recognized".format(self._aggr))

        pos = None  # pos.mean(1).unsqueeze(1)
        x = x.unsqueeze(-1)
        return Data(x=x, pos=pos)

    def __repr__(self):
        return "{}: {} (aggr={}, {})".format(self.__class__.__name__, self.nb_params, self._aggr, self.nn)
