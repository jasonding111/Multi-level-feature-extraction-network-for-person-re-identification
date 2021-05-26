import glob
import re
import os.path as osp
from data.dataset_base import BaseImageDataset


class DukeMTMC_reID(BaseImageDataset):

    dataset_dir = 'DukeMTMC-reID'

    def __init__(self, root='./datasets', verbose=True):
        super(DukeMTMC_reID, self).__init__()
        self.name = "DukeMTMC-reID"
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self.check_before_run(self.dataset_dir, self.train_dir, self.query_dir, self.gallery_dir)

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> DukeMTMC-reID loaded")
            self.print_dataset_info(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_dataset_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_dataset_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_dataset_info(self.gallery)

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            assert 1 <= camid <= 8
            camid -= 1
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset
