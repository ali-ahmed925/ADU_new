import os.path as osp

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
# from ..bas import Datum, DatasetBase
import random
DOMAIN_NAMES = [
    "art_painting",
    "cartoon",
    "photo",
    "sketch"
]

CLASS_NAMES = [
    "dog", "elephant", "giraffe", "guitar", "horse", "house", "person"
]

@DATASET_REGISTRY.register()
class PACSDF(DatasetBase):
    """PACS.

    Statistics:
        - 4 domains: Photo (1,670), Art (2,048), Cartoon
        (2,344), Sketch (3,929).
        - 7 categories: dog, elephant, giraffe, guitar, horse,
        house and person.

    Reference:
        - Li et al. Deeper, broader and artier domain generalization.
        ICCV 2017.
    """

    dataset_dir = "pacs"
    domains = ["art_painting", "cartoon", "photo", "sketch"]
    data_url = "https://drive.google.com/uc?id=1m4X4fROCCXMO0lRLrr6Zz9Vb3974NWhE"
    # the following images contain errors and should be ignored
    _error_paths = ["sketch/dog/n02103406_4068-1.png"]

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.image_dir = osp.join(self.dataset_dir, "images")
        self.split_dir = osp.join(self.dataset_dir, "splits")
        train_domains = ["art_painting", "cartoon", "photo", "sketch"]
        test_domains = ["art_painting", "cartoon", "photo", "sketch"]


        if not osp.exists(self.dataset_dir):
            dst = osp.join(root, "pacs.zip")
            self.download_data(self.data_url, dst, from_gdrive=True)

        self.check_input_domains(
            train_domains, train_domains
        )

        train = self._read_data(train_domains, "train")
        val = self._read_data(train_domains, "crossval")
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
            if split == "all":
                file_train = osp.join(
                    self.split_dir, dname + "_train_kfold.txt"
                )
                impath_label_list = self._read_split_pacs(file_train)
                file_val = osp.join(
                    self.split_dir, dname + "_crossval_kfold.txt"
                )
                impath_label_list += self._read_split_pacs(file_val)
            else:
                file = osp.join(
                    self.split_dir, dname + "_" + split + "_kfold.txt"
                )
                impath_label_list = self._read_split_pacs(file)

            for impath, label in impath_label_list:
                classname = impath.split("/")[-2]
                item = Datum(
                    impath=impath,
                    label=label,
                    domain=domain,
                    classname=classname
                )
                items.append(item)

        return items

    def _read_split_pacs(self, split_file):
        items = []

        with open(split_file, "r") as f:
            lines = f.readlines()

            for line in lines:
                line = line.strip()
                impath, label = line.split(" ")
                if impath in self._error_paths:
                    continue
                impath = osp.join(self.image_dir, impath)
                label = int(label) - 1
                items.append((impath, label))

        return items
