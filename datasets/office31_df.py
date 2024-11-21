import random
import os.path as osp
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from dassl.utils import listdir_nohidden
from dassl.data.datasets import DATASET_REGISTRY
from dassl.data.datasets import Datum, DatasetBase


@DATASET_REGISTRY.register()
class Office31DF(DatasetBase):
    """Office-31.

    Statistics:
        - 4,110 images.
        - 31 classes related to office objects.
        - 3 domains: Amazon, Webcam, Dslr.
        - URL: https://people.eecs.berkeley.edu/~jhoffman/domainadapt/.

    Reference:
        - Saenko et al. Adapting visual category models to
        new domains. ECCV 2010.
    """

    dataset_dir = "office31"
    domains = ["amazon", "webcam", "dslr"]

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)

        # self.check_input_domains(
        #     cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        # )
        train_domains = ["amazon", "webcam", "dslr"]

        train_x, test_x = self._read_data(train_domains)
        # train_u, test_u = self._read_data(cfg.DATASET.TARGET_DOMAINS)
        num_shots = cfg.DATASET.NUM_SHOTS  # 使用する数ショット数を設定
        train_x = self.generate_fewshot_dataset(train_x, num_shots=num_shots, repeat=True, seed=cfg.SEED)

        super().__init__(train_x=train_x,  test=test_x)
    
    def generate_fewshot_dataset(self, *data_sources, num_shots=-1, repeat=False, seed=0):
        random.seed(seed)
        if num_shots < 1:
            if len(data_sources) == 1:
                return data_sources[0]
            return data_sources

        print(f"Creating a {num_shots}-shot dataset with domain consideration")

        output = []

        for data_source in data_sources:
            tracker = self.split_dataset_by_label_and_domain(data_source)  # ドメインごとに分割
            dataset = []

            for (label, domain), items in tracker.items():
                if len(items) >= num_shots:
                    sampled_items = random.sample(items, num_shots)
                else:
                    sampled_items = random.choices(items, k=num_shots) if repeat else items
                dataset.extend(sampled_items)

            output.append(dataset)

        return output[0] if len(output) == 1 else output


    def split_dataset_by_label_and_domain(self, data_source):
        """
        Split the dataset by both label and domain.
        
        Args:
            data_source: List of Datum objects.
        
        Returns:
            A dictionary with keys as (label, domain) tuples and values as lists of Datum objects.
        """
        tracker = {}
        
        for item in data_source:
            key = (item.label, item.domain)  # (label, domain)のタプルをキーとして使用
            if key not in tracker:
                tracker[key] = []
            tracker[key].append(item)
        
        return tracker

    def _read_data(self, input_domains, split_ratio=0.8):
        items = []
        labels = []  # ラベルリスト（StratifiedShuffleSplitで使用）

        for domain, dname in enumerate(input_domains):
            domain_dir = osp.join(self.dataset_dir, dname)
            class_names = listdir_nohidden(domain_dir)
            class_names.sort()

            for label, class_name in enumerate(class_names):
                class_path = osp.join(domain_dir, class_name)
                imnames = listdir_nohidden(class_path)

                for imname in imnames:
                    impath = osp.join(class_path, imname)
                    item = Datum(
                        impath=impath,
                        label=label,
                        domain=domain,
                        classname=class_name
                    )
                    items.append(item)
                    labels.append(label)

        # Stratified 分割
        items = np.array(items)
        labels = np.array(labels)
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=1-split_ratio, random_state=42
        )
        train_idx, test_idx = next(splitter.split(items, labels))

        # 分割結果
        train_items = items[train_idx].tolist()
        test_items = items[test_idx].tolist()

        return train_items, test_items