import os.path as osp


class BaseImageDataset(object):

    def check_before_run(self, dataset_dir, train_dir, query_dir, gallery_dir):
        if not osp.exists(dataset_dir):
            raise RuntimeError("'{0}' is not available".format(dataset_dir))
        if not osp.exists(train_dir):
            raise RuntimeError("'{0}' is not available".format(train_dir))
        if not osp.exists(query_dir):
            raise RuntimeError("'{0}' is not available".format(query_dir))
        if not osp.exists(gallery_dir):
            raise RuntimeError("'{0}' is not available".format(gallery_dir))

    def get_dataset_info(self, data):
        pids, cams = [], []
        for _, pid, camid in data:
            pids += [pid]
            cams += [camid]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        return num_pids, num_imgs, num_cams

    def print_dataset_info(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams = self.get_dataset_info(train)
        num_query_pids, num_query_imgs, num_query_cams = self.get_dataset_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams = self.get_dataset_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")
