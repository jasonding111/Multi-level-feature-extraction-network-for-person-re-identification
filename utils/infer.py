import time
import numpy as np
import torch
from losses.triplet_loss import normalize, cosine_similarity
from utils import visualizer
from utils.metrics import evaluate
from utils.logger import AverageMeter
from utils.reranking import re_ranking


def inference(model, query_loader, gallery_loader, dataset_name):
    ranks = [1, 5, 10, 20]
    RK = True
    save = False
    if RK:
        RK_name = dataset_name + "(RK)" + ".pickle"
        name = dataset_name + ".pickle"
    else:
        name = dataset_name + ".pickle"
    batch_time = AverageMeter()
    model.eval()
    model = model.cuda()
    # model = nn.DataParallel(model).cuda()
    with torch.no_grad():
        qf, q_pids, q_camids, q_imgs = [], [], [], []
        for batch_idx, (imgs, pids, camids) in enumerate(query_loader):
            imgs = imgs.cuda()
            start = time.time()
            features = model(imgs)

            batch_time.update(time.time() - start)

            features = features.data.cpu()

            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)

        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)
        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids, g_imgs = [], [], [], []
        for batch_idx, (imgs, pids, camids) in enumerate(gallery_loader):
            imgs = imgs.cuda()

            start = time.time()
            features = model(imgs)
            batch_time.update(time.time() - start)

            features = features.data.cpu()

            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)

        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)
        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

    qf = normalize(qf)
    gf = normalize(gf)

    # distmat = euclidean_dist(qf, gf)
    distmat = cosine_similarity(qf, gf)
    distmat = distmat.numpy()

    print("Computing CMC and mAP")
    cmc, mAP, all_ap = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

    print("Results ----------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
    print("------------------")
    if save:
        visualizer.save_info("./vis_rank_list", name, mAP, cmc[0], all_ap, distmat, q_pids, g_pids, q_camids, g_camids)
    if RK:
        print("Using global for reranking")
        distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
        print("Computing CMC and mAP for re_ranking")
        cmc, mAP, all_ap = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

        print("Results ----------")
        print("mAP(RK): {:.1%}".format(mAP))
        print("CMC curve(RK)")
        for r in ranks:
            print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
        print("------------------")
        if save:
            visualizer.save_info("./vis_rank_list", RK_name, mAP, cmc[0], all_ap, distmat, q_pids, g_pids, q_camids, g_camids)
