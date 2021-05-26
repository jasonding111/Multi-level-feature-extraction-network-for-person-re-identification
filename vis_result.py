import os

from data.data_loader_build import build_data_loader
from utils.logger import setup_logger
from utils.visualizer import Visualizer
import os.path as osp


def vis_result(args=None):
    from utils import visualizer
    if args is not None:
        if args["win"] == "True":
            windows = True
        else:
            windows = False
        if args["rk"] == "True":
            RK = True
        else:
            RK = False
        data = args["dataset"]
        vis_label = True
        num_vis = int(args["vis_num"])
        # "ascending" or "descending"
        rank_sort = args["rank_sort"]
        label_sort = args["label_sort"]
        vis_max_rank = int(args["rank_num"])
    else:
        data = 'm'
        RK = False
        windows = True
        vis_label = True
        num_vis = 10
        # "ascending" or "descending"
        rank_sort = "descending"
        label_sort = "descending"
        vis_max_rank = 10
    if windows:
        root = ".\\checkpoint"
        vis_output = ".\\img"
    else:
        root = "./checkpoint"
        vis_output = "./img"
    _, _, _, _, _, test_loader,dataset_name = build_data_loader(data)
    if RK:
        name = dataset_name+"(RK)"
    else:
        name = dataset_name
    vis_name = name + ".pickle"
    vis_res = osp.join(root, vis_name)
    if not osp.exists(vis_res):
        raise RuntimeError("'{0}' is not available".format(vis_res))
    res = visualizer.load_info(vis_res)
    print("------------------")
    print("mAP: {:.1%}".format(res["mAP"]))
    print("Rank-1: {:.1%}".format(res["rank1"]))
    print("------------------")
    logger = setup_logger()
    visualizer = Visualizer(test_loader.dataset)
    visualizer.get_model_output(res["all_ap"], res["distmat"], res["q_pids"], res["g_pids"], res["q_camids"],
                                res["g_camids"])
    logger.info("Drawing ROC curve ...")
    visualizer.vis_roc_curve(vis_output, name)

    # logger.info("Saving rank list result ...")
    # query_indices = visualizer.vis_rank_list(vis_output, vis_label, num_vis,
    #                                          rank_sort, label_sort, vis_max_rank)


if __name__ == '__main__':
    vis_result()
