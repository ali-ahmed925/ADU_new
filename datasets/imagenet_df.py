import os
import pickle
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing

from .imagenet import ImageNet
from .imagenet_sketch import ImageNetSketch

DOMAIN_NAMES = ["real", "sketch"]

@DATASET_REGISTRY.register()
class ImageNetDF(DatasetBase):
    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)

        # Initialize original ImageNet dataset
        self.imagenet = ImageNet(cfg)
        original_train = self.imagenet.train_x
        original_test = self.imagenet.val

        # Initialize ImageNetSketch dataset
        self.imagenet_sketch = ImageNetSketch(cfg)
        sketch_train = self.imagenet_sketch.train_x
        sketch_test = self.imagenet_sketch.test

        # Add domain information to the data
        original_train = self.add_domain(original_train, domain="real")
        original_test = self.add_domain(original_test, domain="real")
        sketch_train = self.add_domain(sketch_train, domain="sketch")
        sketch_test = self.add_domain(sketch_test, domain="sketch")

        # Combine datasets
        train = original_train + sketch_train
        test = original_test + sketch_test

        super().__init__(train_x=train, val=test, test=test)

    @staticmethod
    def add_domain(data, domain):
        """Add domain information to the data items."""
        # for item in data:
        #     item.domain = domain
        #     # if item.domain != 0:
        #     #     print(item.domain)
        #     # else :
        #     #     print(item.domain)
        # return data
        new_data = []
        for item in data:
            new_item = Datum(
                impath=item.impath,
                label=item.label,
                classname=item.classname,
                domain = DOMAIN_NAMES.index(domain)
            )
            # new_item.domain = domain  # 新しいオブジェクトに domain を設定
            new_data.append(new_item)
        return new_data