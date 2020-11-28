from abc import ABC, abstractmethod
import math
import torch
from torch_geometric.nn import voxel_grid
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.nn.pool.pool import pool_pos, pool_batch
import torch_points_kernels as tp


class BaseSampler(ABC):
    """If num_to_sample is provided, sample exactly
        num_to_sample points. Otherwise sample floor(pos[0] * ratio) points
    """

    def __init__(self, ratio=None, num_to_sample=None, subsampling_param=None):
        if num_to_sample is not None:
            if (ratio is not None) or (subsampling_param is not None):
                raise ValueError("Can only specify ratio or num_to_sample or subsampling_param, not several !")
            self._num_to_sample = num_to_sample

        elif ratio is not None:
            self._ratio = ratio

        elif subsampling_param is not None:
            self._subsampling_param = subsampling_param

        else:
            raise Exception('At least ["ratio, num_to_sample, subsampling_param"] should be defined')

    def __call__(self, data):
        return self.sample(data)

    def _get_num_to_sample(self, batch_size) -> int:
        if hasattr(self, "_num_to_sample"):
            return self._num_to_sample
        else:
            return math.floor(batch_size * self._ratio)

    def _get_ratio_to_sample(self, batch_size) -> float:
        if hasattr(self, "_ratio"):
            return self._ratio
        else:
            return self._num_to_sample / float(batch_size)

    @abstractmethod
    def sample(self, pos, x=None, batch=None):
        pass


class DenseFPSSampler(BaseSampler):
    """If num_to_sample is provided, sample exactly
        num_to_sample points. Otherwise sample floor(pos[0] * ratio) points
    """

    def sample(self, data, **kwargs):
        """ Sample pos

        Arguments:
            data.pos -- [B, N, 3]

        Returns:
            indexes -- [B, num_sample]
        """
        if len(data.pos.shape) != 3:
            raise ValueError(" This class is for dense data and expects the pos tensor to be of dimension 2")
        idx = tp.furthest_point_sample(data.pos, self._get_num_to_sample(data.pos.shape[1])).to(torch.long)
        data.pos = torch.gather(data.pos, 1, idx.unsqueeze(-1).repeat(1, 1, data.pos.shape[-1]))
        data.x = torch.gather(data.x, 2, idx.unsqueeze(1).unsqueeze(-1).repeat(1, data.x.shape[1], 1, 1))
        return data, idx


class DenseRandomSampler(BaseSampler):
    """If num_to_sample is provided, sample exactly
        num_to_sample points. Otherwise sample floor(pos[0] * ratio) points
        Arguments:
            data.pos -- [B, N, 3]
    """

    def sample(self, data, **kwargs):
        if len(data.pos.shape) != 3:
            raise ValueError(" This class is for dense data and expects the pos tensor to be of dimension 2")
        idx = torch.randint(0, data.pos.shape[1], (data.pos.shape[0], self._get_num_to_sample(data.pos.shape[1]))).to(data.pos.device)
        data.pos = torch.gather(data.pos, 1, idx.unsqueeze(-1).repeat(1, 1, data.pos.shape[-1]))
        data.x = torch.gather(data.x, 2, idx.unsqueeze(1).unsqueeze(-1).repeat(1, data.x.shape[1], 1, 1))
        return data, idx


