import glob
import os.path as osp
from dassl.utils import listdir_nohidden
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing
from dassl.data.datasets.dg import DigitsDG
import random

DOMAIN_NAMES = ["art", "clipart", "product", "real_world"]

@DATASET_REGISTRY.register()
class OfficeHomeDF(DatasetBase):
    """Office-Home.

    Statistics:
        - Around 15,500 images.
        - 65 classes related to office and home objects.
        - 4 domains: Art, Clipart, Product, Real World.
        - URL: http://hemanthdv.org/OfficeHome-Dataset/.

    Reference:
        - Venkateswara et al. Deep Hashing Network for Unsupervised
        Domain Adaptation. CVPR 2017.
    """

    def __init__(self, cfg):
        dataset_dir = "office_home_dg"
        train_domains = ["art", "clipart", "product", "real_world"]
        test_domains = ["art", "clipart", "product", "real_world"]
        data_url = "https://drive.google.com/uc?id=1gkbf_KaxoBws-GWT3XIPZ7BnkqbAxIFa"
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, dataset_dir)
        train, val, test = [], [], []
        if not osp.exists(self.dataset_dir):
            dst = osp.join(root, "office_home_dg.zip")
            self.download_data(data_url, dst, from_gdrive=True)

        # self.check_input_domains(
        #     cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        # )
        for domain in train_domains: #FIXME
            train += read_data(
                self.dataset_dir, [domain], "train"
            )
            val += read_data(
                self.dataset_dir, [domain], "val"
            )
        
        num_shots = cfg.DATASET.NUM_SHOTS  # 使用する数ショット数を設定
        train = self.generate_fewshot_dataset(train, num_shots=num_shots, repeat=True, seed=cfg.SEED)

        for domain in test_domains:
            test += read_data(
                self.dataset_dir, [domain], "val"
            )
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


def read_data(dataset_dir, input_domains, split):

        def _load_data_from_directory(directory):
            folders = listdir_nohidden(directory)
            folders.sort()
            items_ = []

            for label, folder in enumerate(folders):
                impaths = glob.glob(osp.join(directory, folder, "*.jpg"))

                for impath in impaths:
                    items_.append((impath, label))

            return items_

        items = []

        for dname in input_domains:
            if split == "all":
                train_dir = osp.join(dataset_dir, dname, "train")
                impath_label_list = _load_data_from_directory(train_dir)
                val_dir = osp.join(dataset_dir, dname, "val")
                impath_label_list += _load_data_from_directory(val_dir)
            else:
                split_dir = osp.join(dataset_dir, dname, split)
                impath_label_list = _load_data_from_directory(split_dir)

            for impath, label in impath_label_list:
                class_name = impath.split("/")[-2].lower()
                item = Datum(
                    impath=impath,
                    label=label,
                    domain=DOMAIN_NAMES.index(dname),
                    classname=class_name
                )
                items.append(item)

        return items

    
