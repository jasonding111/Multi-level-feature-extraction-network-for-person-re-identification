import torch
from torch.utils.data import DataLoader
from data.dataset_manager import ImageDataset
from data.dukemtmc_reid import DukeMTMC_reID
from data.market1501 import Market1501
from data.samplers import RandomIdentitySampler
from data.transforms_build import build_transforms


def train_collate_fn(batch):
    imgs, pids, _, _, = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids


def test_collate_fn(batch):
    imgs, pids, camids, _ = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids


def build_data_loader(data):
    batch_num = 24
    num_instance = 4
    num_workers = 8

    train_transforms = build_transforms(is_train=True)
    test_transforms = build_transforms(is_train=False)

    if data == 'm':
        dataset = Market1501()
    elif data == 'd':
        dataset = DukeMTMC_reID()
    else:
        print("no dataset")
        quit(0)
    dataset_name = dataset.name
    num_classes = dataset.num_train_pids
    train_loader = DataLoader(
        ImageDataset(dataset.train, train_transforms),
        batch_size=batch_num, sampler=RandomIdentitySampler(dataset.train, batch_num, num_instance),
        num_workers=num_workers, collate_fn=train_collate_fn,
        pin_memory=True
    )
    query_loader = DataLoader(
        ImageDataset(dataset.query, transform=test_transforms),
        batch_size=batch_num, shuffle=False, num_workers=num_workers,
        collate_fn=test_collate_fn,
        pin_memory=True
    )
    gallery_loader = DataLoader(
        ImageDataset(dataset.gallery, transform=test_transforms),
        batch_size=batch_num, shuffle=False, num_workers=num_workers,
        collate_fn=test_collate_fn,
        pin_memory=True
    )
    test_loader = DataLoader(
        ImageDataset(dataset.query + dataset.gallery, transform=test_transforms),
        batch_size=batch_num, shuffle=False, num_workers=num_workers,
        collate_fn=test_collate_fn,
        pin_memory=True
    )
    return train_loader, query_loader, gallery_loader, len(dataset.query), num_classes, test_loader, dataset_name
