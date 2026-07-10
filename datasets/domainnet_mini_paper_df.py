"""
DomainNetMini with the official 126-class splits (splits_mini/).
Matches the paper's experimental setup exactly.
Labels are already 0-125 in the split files.
"""
import os.path as osp
import random

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing


DOMAIN_NAMES = ["clipart", "painting", "real", "sketch"]


@DATASET_REGISTRY.register()
class DomainNetMiniPaperDF(DatasetBase):
    """Mini DomainNet — 4 domains, 126 classes, official splits_mini split.

    Images are read from DomainNet/ using the splits_mini/*.txt files.
    Labels are pre-remapped to 0-125 in the split files.
    """

    dataset_dir = "DomainNet"
    domains = DOMAIN_NAMES

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.split_dir = osp.join(self.dataset_dir, "splits_mini")

        train_domains = DOMAIN_NAMES
        test_domains = DOMAIN_NAMES

        train_x = self._read_data(train_domains, split="train")
        val = self._read_data(test_domains, split="test")
        test = self._read_data(test_domains, split="test")

        num_shots = cfg.DATASET.NUM_SHOTS
        train_x = self._generate_fewshot(train_x, num_shots, seed=cfg.DATASET.SEED)

        super().__init__(train_x=train_x, val=val, test=test)

    def _read_data(self, input_domains, split="train"):
        items = []
        for domain_idx, dname in enumerate(input_domains):
            split_file = osp.join(self.split_dir, f"{dname}_{split}.txt")
            with open(split_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    impath, label = line.split(" ")
                    classname = impath.split("/")[1]
                    impath = osp.join(self.dataset_dir, impath)
                    item = Datum(
                        impath=impath,
                        label=int(label),
                        domain=domain_idx,
                        classname=classname,
                    )
                    items.append(item)
        return items

    def _generate_fewshot(self, data, num_shots, seed=0):
        if num_shots < 1:
            return data
        random.seed(seed)
        # group by (label, domain)
        tracker = {}
        for item in data:
            key = (item.label, item.domain)
            tracker.setdefault(key, []).append(item)
        out = []
        for items in tracker.values():
            if len(items) >= num_shots:
                out.extend(random.sample(items, num_shots))
            else:
                out.extend(random.choices(items, k=num_shots))
        return out
