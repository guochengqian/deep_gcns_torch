import __init__
from tqdm import tqdm
import numpy as np
import torch_geometric.datasets as GeoData
from torch_geometric.data import DenseDataLoader
import torch_geometric.transforms as T
from config import OptInit
from architecture_mapgap import DenseDeepGCN
from utils.ckpt_util import load_pretrained_models
import logging
from MADGap import *


def main():
    opt = OptInit().get_args()

    logging.info('===> Creating dataloader...')
    test_dataset = GeoData.S3DIS(opt.data_dir, opt.area, train=False, pre_transform=T.NormalizeScale())
    test_loader = DenseDataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0)
    opt.n_classes = test_loader.dataset.num_classes
    if opt.no_clutter:
        opt.n_classes -= 1

    logging.info('===> Loading the network ...')
    model = DenseDeepGCN(opt).to(opt.device)
    model, opt.best_value, opt.epoch = load_pretrained_models(model, opt.pretrained_model, opt.phase)

    logging.info('===> Start Evaluation ...')
    test(model, test_loader, opt)


def test(model, loader, opt):
    Is = np.empty((len(loader), opt.n_classes))
    Us = np.empty((len(loader), opt.n_classes))
    n_layers = opt.n_blocks + 1
    n_batch = len(loader)
    n_classes = opt.n_classes

    mean_MAD_mat = torch.zeros(n_batch, n_layers)
    MADFarthest_mat = torch.zeros(n_batch, n_layers)
    MADClosest_mat = torch.zeros(n_batch, n_layers)

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(tqdm(loader)):
            data = data.to(opt.device)
            gt = data.y

            # forward
            data.x = torch.cat((data.pos.transpose(2, 1).unsqueeze(3), data.x.transpose(2, 1).unsqueeze(3)), 1)
            out, feats = model(data.pos, data.x)

            # mIoU
            pred = out.max(dim=1)[1]
            pred_np = pred.cpu().numpy()
            target_np = gt.cpu().numpy()
            for cl in range(opt.n_classes):
                cur_gt_mask = (target_np == cl)
                cur_pred_mask = (pred_np == cl)
                I = np.sum(np.logical_and(cur_pred_mask, cur_gt_mask), dtype=np.float32)
                U = np.sum(np.logical_or(cur_pred_mask, cur_gt_mask), dtype=np.float32)
                Is[i, cl] = I
                Us[i, cl] = U

            # MADGap
            label = gt.view(-1)

            # removes the dominant classes such ceiling floor and wall. Class imbalance
            mask = (label != 0) * (label != 1) * (label != 2)
            # for j in range(n_layers):
            j = n_layers-1
            feat = feats[j].squeeze(-1).permute(0, 2, 1)    # feat: B, N, C
            mean_MAD, MADFarthest, MADClosest = batchwise_MADGap(feat, 16, j + 1)
            MADFarthest = MADFarthest.view(-1)
            MADClosest = MADClosest.view(-1)
            mean_MAD_mat[i, j] = mean_MAD
            MADFarthest_mat[i, j] = nanmean(MADFarthest[mask])
            MADClosest_mat[i, j] = nanmean(MADClosest[mask])
    MADGap = nanmean(MADFarthest_mat - MADClosest_mat, dim=0, keepdims=False)
    torch.save(MADGap, '{}/{}'.format(opt.res_dir, 'madgap.pt'))
    logging.info(MADGap)

    mean_MAD = torch.mean(mean_MAD_mat, dim=0, keepdim=False)
    logging.info(mean_MAD)
    ious = np.divide(np.sum(Is, 0), np.sum(Us, 0))
    ious[np.isnan(ious)] = 1
    for cl in range(opt.n_classes):
        logging.info("===> mIOU for class {}: {}".format(cl, ious[cl]))
    logging.info("===> mIOU is {}".format(np.mean(ious)))


if __name__ == '__main__':
    main()
