import os

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden

from .imagenet import ImageNet
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

@DATASET_REGISTRY.register()
class ImageNetSketch(DatasetBase):
    """ImageNet-Sketch.

    This dataset is used for testing only.
    """

    dataset_dir = "imagenet-sketch"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser("/home/gotoyuta/lab/Dataset"))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")

        text_file = os.path.join(self.dataset_dir, "classnames.txt")
        classnames = ImageNet.read_classnames(text_file)

        num_shots = cfg.DATASET.NUM_SHOTS

        train, test = self.read_data(classnames)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)
        super().__init__(train_x=train, val=test, test=test)

    def read_data(self, classnames, split_ratio=0.8):
        image_dir = self.image_dir
        folders = listdir_nohidden(image_dir, sort=True)
        items = []
        labels = []
        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(image_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(image_dir, folder, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)
                labels.append(label)
        
        items = np.array(items)
        labels = np.array(labels)
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=1-split_ratio, random_state=42
        )
        train_idx, test_idx = next(splitter.split(items, labels))

        train_items = items[train_idx].tolist()
        test_items = items[test_idx].tolist()

        return train_items, test_items
