import torch
from gcn_lib.dense import (BasicConv, MLP, GraphConv2d, PlainDynBlock2d, ResDynBlock2d,
                           DenseDynBlock2d, DenseDilatedKnnGraph, DynConv2d)
from torch.nn import Sequential as Seq
from gcn_lib.dense.sampling import DenseRandomSampler, DenseFPSSampler
from gcn_lib.dense.upsampling import DenseFPModule
from torch_geometric.data import Data
import warnings


class DenseDeepGCN(torch.nn.Module):
    def __init__(self, opt):
        super(DenseDeepGCN, self).__init__()
        channels = opt.n_filters
        k = opt.k
        act = opt.act
        norm = opt.norm
        bias = opt.bias
        epsilon = opt.epsilon
        stochastic = opt.use_stochastic
        conv = opt.conv
        c_growth = channels
        self.n_blocks = opt.n_blocks

        self.knn = DenseDilatedKnnGraph(k, 1, stochastic, epsilon)
        self.head = GraphConv2d(opt.in_channels, channels, conv, act, norm, bias)

        if opt.block.lower() == 'res':
            if opt.use_dilation:
                self.backbone = Seq(*[ResDynBlock2d(channels, k, 1+i, conv, act, norm, bias, stochastic, epsilon)
                                      for i in range(self.n_blocks-1)])
            else:
                self.backbone = Seq(*[ResDynBlock2d(channels, k, 1, conv, act, norm, bias, stochastic, epsilon)
                                      for i in range(self.n_blocks-1)])

            fusion_dims = int(channels + c_growth * (self.n_blocks - 1))

        elif opt.block.lower() == 'dense':
            if opt.use_dilation:
                self.backbone = Seq(*[DenseDynBlock2d(channels+c_growth*i, c_growth, k, 1+i, conv, act,
                                                      norm, bias, stochastic, epsilon)
                                      for i in range(self.n_blocks-1)])
            else:
                self.backbone = Seq(*[DenseDynBlock2d(channels+c_growth*i, c_growth, k, 1, conv, act,
                                                      norm, bias, stochastic, epsilon)
                                      for i in range(self.n_blocks-1)])
            fusion_dims = int(
                (channels + channels + c_growth * (self.n_blocks - 1)) * self.n_blocks // 2)
        else:
            if opt.use_dilation:
                warnings.warn("use_dilation is set to True for PlainGCN. "
                              "Make sure this is what you want. "
                              "if not, please use no_stochastic, no_dilation")
            if stochastic:
                warnings.warn("use_stochastic is set to True for PlainGCN. "
                              "Make sure this is what you want. "
                              "if not, please use no_stochastic, no_dilation")
            if opt.use_dilation:
                self.backbone = Seq(*[PlainDynBlock2d(channels, k, 1+i, conv, act, norm,
                                                      bias, stochastic, epsilon)
                                      for i in range(self.n_blocks - 1)])
            else:
                self.backbone = Seq(*[PlainDynBlock2d(channels, k, 1, conv, act, norm,
                                                      bias, stochastic, epsilon)
                                      for i in range(self.n_blocks - 1)])
            fusion_dims = int(channels + c_growth * (self.n_blocks - 1))

        self.fusion_block = BasicConv([fusion_dims, 1024], act, norm, bias)
        self.prediction = Seq(*[BasicConv([fusion_dims+1024, 512], act, norm, bias),
                                BasicConv([512, 256], act, norm, bias),
                                torch.nn.Dropout(p=opt.dropout),
                                BasicConv([256, opt.n_classes], None, None, bias)])

    def forward(self, pos, x):
        feats = [self.head(x, self.knn(x[:, 0:3]))]
        for i in range(self.n_blocks-1):
            feats.append(self.backbone[i](feats[-1]))

        last_feat = feats[-1]
        feats = torch.cat(feats, dim=1)

        fusion = torch.max_pool2d(self.fusion_block(feats), kernel_size=[feats.shape[2], feats.shape[3]])
        fusion = torch.repeat_interleave(fusion, repeats=feats.shape[2], dim=2)
        out = self.prediction(torch.cat((fusion, feats), dim=1))
        # featmaps.append(out.clone())
        # return out.squeeze(-1), featmaps
        return out.squeeze(-1), last_feat
        # return out.squeeze(-1)


class DeepGCNUNet(torch.nn.Module):
    def __init__(self, opt):
        super(DeepGCNUNet, self).__init__()
        channels = opt.n_filters
        k = opt.k
        act = opt.act
        norm = opt.norm
        bias = opt.bias
        epsilon = opt.epsilon
        stochastic = opt.stochastic
        conv = opt.conv
        c_growth = channels
        self.n_blocks = opt.n_blocks

        self.down_layers = opt.down_layers
        self.down_interval = opt.n_blocks // self.down_layers

        # assert opt.block.lower() == "res" # only support ResGCN-UNet

        self.knn = DenseDilatedKnnGraph(k, 1, stochastic, epsilon)
        self.head = GraphConv2d(opt.in_channels, channels, conv, act, norm, bias)

        if opt.sampler == 'random':
            self.sampler = DenseRandomSampler(ratio=0.5)
        elif opt.sampler == 'fps':
            self.sampler = DenseFPSSampler(ratio=0.5)
        else:
            raise NotImplementedError("{} is not implemented".format(opt.sampler))

        # architecture is ResGCN
        self.backbone = torch.nn.ModuleList()
        for i in range(self.n_blocks-1):
            if (i+1) % self.down_interval == 0:
                self.backbone.append(DynConv2d(channels, channels*2, k, 1, conv, act, norm, bias, stochastic, epsilon))
                channels = channels*2
            else:
                self.backbone.append(ResDynBlock2d(channels, k, 1, conv, act, norm, bias, stochastic, epsilon))

        fusion_dims = channels
        self.up_modules = torch.nn.ModuleList()
        for i in range(self.down_layers):
            self.up_modules.append(DenseFPModule([channels + channels//2, channels//2]))
            channels = channels // 2
            fusion_dims += channels

        self.prediction = Seq(*[MLP([64, 64], act, norm, bias),
                                MLP([64, 32], act, norm, bias),
                                torch.nn.Dropout(p=opt.dropout),
                                MLP([32, opt.n_classes], None, None, bias)])

    def forward(self, pos, x):
        data = Data(pos=pos, x=x)
        feat_h = self.head(data.x, self.knn(data.x[:, 0:3]))
        data.x = feat_h

        # feat_maps = [feat_h.clone()]
        stack_down = [data.clone()]
        for i in range(self.n_blocks-1):
            feat_h = self.backbone[i](data.x)
            data.x = feat_h
            # feat_maps.append(feat_h.clone())
            if (i+1) % self.down_interval == 0:
                data, idx = self.sampler(data.clone())
                stack_down.append(data.clone())

        feats = []
        stack_down.pop()
        for i in range(self.down_layers-1):
            data = self.up_modules[i]((data, stack_down.pop()))
            feats.append(data.x.clone())
        return self.prediction(data.x).squeeze(-1)


if __name__ == "__main__":
    import argparse
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    batch_size = 2
    N = 1024
    device = 'cuda'

    parser = argparse.ArgumentParser(description='PyTorch implementation of Deep GCN For semantic segmentation')
    parser.add_argument('--in_channels', default=9, type=int, help='input channels (default:9)')
    parser.add_argument('--n_classes', default=13, type=int, help='num of segmentation classes (default:13)')
    parser.add_argument('--k', default=20, type=int, help='neighbor num (default:16)')
    parser.add_argument('--block', default='res', type=str, help='graph backbone block type {plain, res, dense}')
    parser.add_argument('--conv', default='edge', type=str, help='graph conv layer {edge, mr}')
    parser.add_argument('--act', default='relu', type=str, help='activation layer {relu, prelu, leakyrelu}')
    parser.add_argument('--norm', default='batch', type=str, help='{batch, instance} normalization')
    parser.add_argument('--bias', default=True, type=bool, help='bias of conv layer True or False')
    parser.add_argument('--n_filters', default=64, type=int, help='number of channels of deep features')
    parser.add_argument('--n_blocks', default=7, type=int, help='number of basic blocks')
    parser.add_argument('--dropout', default=0.5, type=float, help='ratio of dropout')
    parser.add_argument('--epsilon', default=0.2, type=float, help='stochastic epsilon for gcn')
    parser.add_argument('--stochastic', default=False, type=bool, help='stochastic for gcn, True or False')
    args = parser.parse_args()

    pos = torch.rand((batch_size, N, 3), dtype=torch.float).to(device)
    x = torch.rand((batch_size, N, 6), dtype=torch.float).to(device)

    inputs = torch.cat((pos, x), 2).transpose(1, 2).unsqueeze(-1)

    # net = DGCNNSegDense().to(device)
    net = DenseDeepGCN(args).to(device)
    print(net)
    out = net(inputs)
    print(out.shape)
