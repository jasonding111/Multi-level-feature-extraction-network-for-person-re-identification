import pickle
import torchvision.transforms as T
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from scipy.stats import norm
from sklearn import metrics
from data.dataset_manager import read_image
from utils.file_io import PathManager


class Visualizer:

    def __init__(self, dataset):
        self.dataset = dataset

    def get_model_output(self, all_ap, dist, q_pids, g_pids, q_camids, g_camids):
        self.all_ap = all_ap
        self.dist = dist
        self.sim = 1 - dist
        self.q_pids = q_pids
        self.g_pids = g_pids
        self.q_camids = q_camids
        self.g_camids = g_camids

        self.indices = np.argsort(dist, axis=1)
        self.matches = (g_pids[self.indices] == q_pids[:, np.newaxis]).astype(np.int32)

        self.num_query = len(q_pids)

    def get_matched_result(self, q_index):
        q_pid = self.q_pids[q_index]
        q_camid = self.q_camids[q_index]

        order = self.indices[q_index]
        remove = (self.g_pids[order] == q_pid) & (self.g_camids[order] == q_camid)
        keep = np.invert(remove)
        cmc = self.matches[q_index][keep]
        sort_idx = order[keep]
        return cmc, sort_idx

    def save_rank_result(self, query_indices, output, max_rank=5, vis_label=False, label_sort='ascending',
                         actmap=False):
        if vis_label:
            fig, axes = plt.subplots(2, max_rank + 1, figsize=(3 * max_rank, 12))
        else:
            fig, axes = plt.subplots(1, max_rank + 1, figsize=(3 * max_rank, 6))
        transform = T.Resize([256, 128])
        windows = False
        for cnt, q_idx in enumerate(tqdm.tqdm(query_indices)):
            all_imgs = []
            cmc, sort_idx = self.get_matched_result(q_idx)
            q_img, q_pid, q_camid, q_img_path = self.dataset[q_idx]
            # query_img = q_img
            q_img = read_image(q_img_path)
            q_img = transform(q_img)
            cam_id = q_camid + 1
            if windows:
                query_name = q_img_path.split('\\')[-1]
            else:
                query_name = q_img_path.split('/')[-1]
            # all_imgs.append(q_img)
            # query_img = np.rollaxis(np.asarray(query_img.numpy(), dtype=np.uint8), 0, 3)
            plt.clf()
            ax = fig.add_subplot(1, max_rank + 1, 1)
            ax.imshow(q_img)
            ax.set_title('{:.4f}/cam{}'.format(self.all_ap[q_idx], cam_id),fontsize= 25)
            ax.axis("off")
            for i in range(max_rank):
                if vis_label:
                    ax = fig.add_subplot(2, max_rank + 1, i + 2)
                else:
                    ax = fig.add_subplot(1, max_rank + 1, i + 2)
                g_idx = self.num_query + sort_idx[i]
                g_img, g_pid, g_camid, g_img_path = self.dataset[g_idx]
                gallery_img = g_img
                g_img = read_image(g_img_path)
                g_img = transform(g_img)
                cam_id = g_camid + 1
                # if windows:
                #     g_name = g_img_path.split('\\')[-1]
                # else:
                #     g_name = g_img_path.split('/')[-1]
                gallery_img = np.rollaxis(np.asarray(gallery_img, dtype=np.uint8), 0, 3)
                if cmc[i] == 1:
                    label = 'True'
                    ax.add_patch(plt.Rectangle(xy=(0, 0), width=gallery_img.shape[1] - 1,
                                               height=gallery_img.shape[0] - 1, edgecolor=(1, 0, 0),
                                               fill=False, linewidth=5))
                else:
                    label = 'False'
                    ax.add_patch(plt.Rectangle(xy=(0, 0), width=gallery_img.shape[1] - 1,
                                               height=gallery_img.shape[0] - 1,
                                               edgecolor=(0, 0, 1), fill=False, linewidth=5))
                ax.imshow(g_img)
                ax.set_title(f'{label}/cam{cam_id}',fontsize= 25)
                ax.axis("off")
            if vis_label:
                label_indice = np.where(cmc == 1)[0]
                if label_sort == "ascending": label_indice = label_indice[::-1]
                label_indice = label_indice[:max_rank]
                for i in range(max_rank):
                    if i >= len(label_indice): break
                    j = label_indice[i]
                    g_idx = self.num_query + sort_idx[j]
                    g_img, g_pid, g_camid, g_img_path = self.dataset[g_idx]
                    gallery_img = g_img
                    g_img = read_image(g_img_path)
                    g_img = transform(g_img)
                    cam_id = g_camid + 1
                    # if windows:
                    #     g_name = g_img_path.split('\\')[-1]
                    # else:
                    #     g_name = g_img_path.split('/')[-1]
                    gallery_img = np.rollaxis(np.asarray(gallery_img, dtype=np.uint8), 0, 3)
                    ax = fig.add_subplot(2, max_rank + 1, max_rank + 3 + i)
                    ax.add_patch(plt.Rectangle(xy=(0, 0), width=gallery_img.shape[1] - 1,
                                               height=gallery_img.shape[0] - 1,
                                               edgecolor=(1, 0, 0),
                                               fill=False, linewidth=5))
                    ax.imshow(g_img)
                    # ax.set_title(f'{self.sim[q_idx, sort_idx[j]]:.3f}/cam{cam_id}',fontsize= 20)
                    ax.set_title(f'cam{cam_id}', fontsize=25)
                    ax.axis("off")

            plt.tight_layout()
            filepath = os.path.join(output, "{}".format(query_name))
            fig.savefig(filepath)

    def vis_rank_list(self, output, vis_label, num_vis=100, rank_sort="ascending", label_sort="ascending", max_rank=5,
                      actmap=False):
        r"""Visualize rank list of query instance
        Args:
            output (str): a directory to save rank list result.
            vis_label (bool): if visualize label of query
            num_vis (int):
            rank_sort (str): save visualization results by which order,
                if rank_sort is ascending, AP from low to high, vice versa.
            label_sort (bool):
            max_rank (int): maximum number of rank result to visualize
            actmap (bool):
        """
        assert rank_sort in ['ascending', 'descending'], "{} not match [ascending, descending]".format(rank_sort)

        query_indices = np.argsort(self.all_ap)
        if rank_sort == 'descending': query_indices = query_indices[::-1]

        query_indices = query_indices[:num_vis]
        self.save_rank_result(query_indices, output, max_rank, vis_label, label_sort, actmap)

    def vis_roc_curve(self, output, name):
        PathManager.mkdirs(output)
        pos, neg = [], []
        for i, q in enumerate(self.q_pids):
            cmc, sort_idx = self.get_matched_result(i)  # remove same id in same camera
            ind_pos = np.where(cmc == 1)[0]
            q_dist = self.dist[i]
            pos.extend(q_dist[sort_idx[ind_pos]])

            ind_neg = np.where(cmc == 0)[0]
            neg.extend(q_dist[sort_idx[ind_neg]])

        scores = np.hstack((pos, neg))
        labels = np.hstack((np.zeros(len(pos)), np.ones(len(neg))))

        fpr, tpr, thresholds = metrics.roc_curve(labels, scores)
        roc_name = name + "_roc.jpg"
        self.plot_roc_curve(fpr, tpr)
        filepath = os.path.join(output, roc_name)
        plt.savefig(filepath)
        # dist_name = name + "_pos_neg_dist.jpg"
        # self.plot_distribution(pos, neg)
        # filepath = os.path.join(output, dist_name)
        # plt.savefig(filepath)
        return fpr, tpr, pos, neg

    @staticmethod
    def plot_roc_curve(fpr, tpr, name='MFEN', fig=None):
        # if fig is None:
        #     fig = plt.figure()
        #     plt.semilogx(np.arange(0, 1, 0.01), np.arange(0, 1, 0.01), 'r', linestyle='--', label='Random guess')
        # plt.semilogx(fpr, tpr, color=(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)),
        #              label='ROC curve with {}'.format(name))
        # plt.title('Receiver Operating Characteristic')
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.legend(loc='best')
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        if fig is None:
            fig = plt.figure()
            plt.semilogx(np.arange(0, 1, 0.01), np.arange(0, 1, 0.01), 'r', linestyle='--', label='随机预测')
        plt.semilogx(fpr, tpr, color=(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)),
                     label='多级特征提取网络'.format(name))
        plt.title('受试者工作特征曲线')
        plt.xlabel('误识率')
        plt.ylabel('命中率')
        plt.legend(loc='best')
        return fig

    @staticmethod
    def plot_distribution(pos, neg, name='MFEN', fig=None):
        if fig is None:
            fig = plt.figure()
        pos_color = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
        n, bins, _ = plt.hist(pos, bins=80, alpha=0.7, density=True,
                              color=pos_color,
                              label='positive with {}'.format(name))
        mu = np.mean(pos)
        sigma = np.std(pos)
        y = norm.pdf(bins, mu, sigma)  # fitting curve
        plt.plot(bins, y, color=pos_color)  # plot y curve

        neg_color = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
        n, bins, _ = plt.hist(neg, bins=80, alpha=0.5, density=True,
                              color=neg_color,
                              label='negative with {}'.format(name))
        mu = np.mean(neg)
        sigma = np.std(neg)
        y = norm.pdf(bins, mu, sigma)  # fitting curve
        plt.plot(bins, y, color=neg_color)  # plot y curve

        plt.xticks(np.arange(0, 1.5, 0.1))
        plt.title('positive and negative pairs distribution')
        plt.legend(loc='best')
        return fig


def save_info(output, name, mAP, rank1, all_ap, distmat, q_pids, g_pids, q_camids, g_camids):
    results = {
        "mAP": mAP,
        "rank1": rank1,
        "all_ap": all_ap,
        "distmat": distmat,
        "q_pids": q_pids,
        "g_pids": g_pids,
        "q_camids": q_camids,
        "g_camids": g_camids,
    }
    with open(os.path.join(output, name), "wb") as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_info(path):
    with open(path, 'rb') as handle: res = pickle.load(handle)
    return res
