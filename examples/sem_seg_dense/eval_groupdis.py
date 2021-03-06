import __init__
from tqdm import tqdm
import numpy as np
import torch
import torch_geometric.datasets as GeoData
from torch_geometric.data import DenseDataLoader
import torch_geometric.transforms as T
from config import OptInit
from architecture_mapgap import DenseDeepGCN
from utils.ckpt_util import load_pretrained_models
import logging
from MI import mi_kde


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


def dis_cluster(logit, label, num_classes):

    X_labels = []
    X_labels_sum = []

    n_clss = len(np.unique(label))
    if n_clss == 1:
        return 1, 1
    else:
        for i in range(num_classes):
            X_label = logit[label == i]

            h_norm = np.sum(np.square(X_label), axis=1, keepdims=True)
            h_norm[h_norm == 0.] = 1e-3
            h_norm = np.sqrt(h_norm)

            X_label = X_label / h_norm
            X_labels.append(X_label)
            X_labels_sum.append(np.sum(np.square(X_label), axis=1, keepdims=True))

        dis_intra = 0.
        for i in range(num_classes):
            x2 = X_labels_sum[i]
            if len(x2):  # avoid empty list
                dists = x2 + x2.T - 2 * np.matmul(X_labels[i], X_labels[i].T)
                dis_intra += np.mean(dists)
        dis_intra /= n_clss

        dis_inter = 0.
        for i in range(num_classes-1):
            x2_i = X_labels_sum[i]
            for j in range(i+1, num_classes):
                x2_j = X_labels_sum[j]
                if len(x2_i) and len(x2_j):
                    dists = x2_i + x2_j.T - 2 * np.matmul(X_labels[i], X_labels[j].T)
                    dis_inter += np.mean(dists)
        num_inter = float((n_clss - 1) * n_clss / 2)
        dis_inter /= num_inter

        return dis_intra, dis_inter


def test(model, loader, opt):
    Is = np.empty((len(loader), opt.n_classes))
    Us = np.empty((len(loader), opt.n_classes))

    group_dists = []
    ins_dists = []

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(tqdm(loader)):
            data = data.to(opt.device)
            gt = data.y

            # forward
            data.x = torch.cat((data.pos.transpose(2, 1).unsqueeze(3), data.x.transpose(2, 1).unsqueeze(3)), 1)
            # out = model(data.pos, data.x)
            out, last_feat = model(data.pos, data.x)

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

            # Group Distance Ratio
            label = gt.cpu().numpy().reshape(-1)
            C = last_feat.shape[1]
            logit = last_feat.squeeze(-1).cpu().numpy().transpose(0, 2, 1).reshape(-1, C)  # 64 is the channels.
            dis_intra, dis_inter = dis_cluster(logit, label, num_classes=opt.n_classes)
            dis_ratio = dis_inter / dis_intra
            dis_ratio = 1. if np.isnan(dis_ratio) else dis_ratio
            group_dists.append(dis_ratio)

            # Instance Distance
            ins_dist = mi_kde(logit, data.x.cpu().numpy().squeeze(-1).transpose(0,2,1).reshape(-1, 9), var=0.1)
            ins_dists.append(ins_dist)

    group_dist = np.nanmean(group_dists)
    logging.info(f"The Group Distance is {group_dist}")

    # show the instance distance 
    ins_dist = np.nanmean(ins_dists)
    logging.info(f"The Instance Distance is {ins_dist}")

    ious = np.divide(np.sum(Is, 0), np.sum(Us, 0))
    ious[np.isnan(ious)] = 1
    for cl in range(opt.n_classes):
        logging.info("===> mIOU for class {}: {}".format(cl, ious[cl]))
    logging.info("===> mIOU is {}".format(np.mean(ious)))


if __name__ == '__main__':
    main()
