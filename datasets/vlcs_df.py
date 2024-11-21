import glob
import os.path as osp

from dassl.utils import listdir_nohidden
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
import random

@DATASET_REGISTRY.register()
class VLCSDF(DatasetBase):
    """VLCS.

    Statistics:
        - 4 domains: CALTECH, LABELME, PASCAL, SUN
        - 5 categories: bird, car, chair, dog, and person.

    Reference:
        - Torralba and Efros. Unbiased look at dataset bias. CVPR 2011.
    """

    dataset_dir = "VLCS"
    domains = ["caltech", "labelme", "pascal", "sun"]
    data_url = "https://drive.google.com/uc?id=1r0WL5DDqKfSPp9E3tRENwHaXNs1olLZd"

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.classname = ["bird", "car", "chair", "dog", "person"]
        # if not osp.exists(self.dataset_dir):
        #     dst = osp.join(root, "vlcs.zip")
        #     self.download_data(self.data_url, dst, from_gdrive=True)

        # self.check_input_domains(
        #     cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        # )
        train_domains = ["caltech", "labelme", "pascal", "sun"]
        test_domains = ["caltech", "labelme", "pascal", "sun"]
        train = self._read_data(train_domains, "train")
        val = self._read_data(test_domains, "test")
        test = self._read_data(test_domains, "test")

        num_shots = cfg.DATASET.NUM_SHOTS  # 使用する数ショット数を設定
        train = self.generate_fewshot_dataset(train, num_shots=num_shots, repeat=True, seed=cfg.SEED)

        super().__init__(train_x=train, val=val, test=test)
    
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

    def _read_data(self, input_domains, split):
        items = []

        for domain, dname in enumerate(input_domains):
            dname = dname.upper()
            path = osp.join(self.dataset_dir, dname, split)
            folders = listdir_nohidden(path)
            folders.sort()

            for label, folder in enumerate(folders):
                impaths = glob.glob(osp.join(path, folder, "*.jpg"))

                for impath in impaths:
                    item = Datum(impath=impath, label=label, domain=domain, classname=self.classname[label])
                    items.append(item)

        return items
