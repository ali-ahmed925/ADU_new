import os.path as osp

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing
from dassl.data.datasets.dg import DigitsDG

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
            train += DigitsDG.read_data(
                self.dataset_dir, [domain], "train"
            )
            val += DigitsDG.read_data(
                self.dataset_dir, [domain], "val"
            )
        for domain in test_domains:
            test += DigitsDG.read_data(
                self.dataset_dir, [domain], "all"
            )

        super().__init__(train_x=train, val=val, test=test)
