import os
import pickle

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
# from dassl.utils import mkdir_if_missing
from dassl.utils import read_json, write_json, mkdir_if_missing
from typing import List
import os.path as osp
import random

DOMAIN_NAMES = [
    "clipart", 
    "infograph",
    "painting",
    "quickdraw",
    "real",
    "sketch"
    ]
CLASS_NAMES = ['airplane', 'anvil', 'apple', 'arm', 'asparagus', 'backpack', 'banana', 'bandage', 'barn', 'baseball_bat', 'basket', 'bathtub', 'beach', 'bed', 'bee', 'bicycle', 'binoculars', 'bird', 'birthday_cake', 'blackberry', 'blueberry', 'boomerang', 'bowtie', 'bracelet', 'bread', 'bridge', 'broccoli', 'bus', 'bush', 'cake', 'calculator', 'camera', 'camouflage', 'cannon', 'car', 'carrot', 'cat', 'ceiling_fan', 'chandelier', 'church', 'circle', 'clock', 'coffee_cup', 'compass', 'computer', 'cooler', 'couch', 'crayon', 'crocodile', 'cup', 'diamond', 'dog', 'dolphin', 'donut', 'door', 'dragon', 'dresser', 'drill', 'dumbbell', 'elbow', 'eye', 'face', 'fan', 'feather', 'fire_hydrant', 'firetruck', 'fish', 'flamingo', 'flip_flops', 'floor_lamp', 'flying_saucer', 'foot', 'frog', 'goatee', 'golf_club', 'guitar', 'hamburger', 'hand', 'harp', 'headphones', 'hedgehog', 'helicopter', 'helmet', 'hexagon', 'hot_air_balloon', 'hot_tub', 'house', 'hurricane', 'ice_cream', 'key', 'lantern', 'leaf', 'light_bulb', 'line', 'lion', 'lipstick', 'lollipop', 'mailbox', 'marker', 'megaphone', 'monkey', 'mosquito', 'mountain', 'nail', 'ocean', 'oven', 'paintbrush', 'palm_tree', 'paper_clip', 'parachute', 'parrot', 'peanut', 'pear', 'peas', 'pencil', 'penguin', 'piano', 'pillow', 'police_car', 'popsicle', 'power_outlet', 'radio', 'rain', 'rainbow', 'rake', 'rhinoceros', 'school_bus', 'scorpion', 'see_saw', 'sheep', 'shorts', 'skateboard', 'skull', 'skyscraper', 'sleeping_bag', 'snail', 'snake', 'snowflake', 'sock', 'speedboat', 'spreadsheet', 'squiggle', 'squirrel', 'star', 'stop_sign', 'stove', 'strawberry', 'streetlight', 'string_bean', 'submarine', 'sun', 'swan', 'sweater', 'swing_set', 'sword', 'table', 'telephone', 'tennis_racquet', 'The_Eiffel_Tower', 'The_Mona_Lisa', 'tiger', 'toaster', 'toe', 'toilet', 'traffic_light', 'train', 'triangle', 'trombone', 'trumpet', 'van', 'washing_machine', 'waterslide', 'wheel', 'windmill', 'wine_bottle', 'wristwatch', 'yoga', 'zigzag', 'raccoon', 'lighter', 'pig', 'alarm_clock', 'animal_migration', 'hockey_puck', 'cookie', 'rollerskates', 'jacket', 'hospital', 'fork', 'ladder', 'keyboard', 'octagon', 'belt', 'kangaroo', 'mushroom', 'crown', 'roller_coaster', 'hourglass', 'pineapple', 'garden_hose', 'candle', 'bench', 'owl', 'knee', 'horse', 'cow', 'chair', 'potato', 'garden', 'jail', 'pants', 'duck', 'canoe', 'camel', 'clarinet', 'brain', 'ant', 'remote_control', 'beard', 'mug', 'diving_board', 'pizza', 'mouse', 'book', 'basketball', 'sandwich', 'picture_frame', 'lobster', 'rabbit', 'pool', 'sailboat', 'broom', 'river', 'bucket', 'hammer', 'angel', 'nose', 'hockey_stick', 'cello', 'house_plant', 'cloud', 'panda', 'finger', 'purse', 'cactus', 'axe', 'microwave', 'bulldozer', 'hat', 'crab', 'motorbike', 'elephant', 'eyeglasses', 'butterfly', 'cruise_ship', 'grass', 'ear', 'moustache', 'fireplace', 'hot_dog', 'flashlight', 'passport', 'ambulance', 'campfire', 'pickup_truck', 'lightning', 'dishwasher', 'baseball', 'mermaid', 'microphone', 'giraffe', 'frying_pan', 'calendar', 'flower', 'cell_phone', 'moon', 'bear', 'fence', 'envelope', 'leg', 'octopus', 'eraser', 'bat', 'lighthouse', 'rifle', 'postcard', 'castle', 'onion', 'knife', 'necklace', 'pond', 'paint_can', 'drums', 'grapes', 'aircraft_carrier', 'pliers', 'map', 'mouth', 'laptop', 'matches', 'bottlecap', 'saw', 'saxophone', 'scissors', 'screwdriver', 'sea_turtle', 'shark', 'shoe', 'shovel', 'sink', 'smiley_face', 'snorkel', 'snowman', 'soccer_ball', 'spider', 'spoon', 'square', 'stairs', 'steak', 'stereo', 'stethoscope', 'stitches', 'suitcase', 'syringe', 'teapot', 'teddy-bear', 'television', 'tent', 'The_Great_Wall_of_China', 'tooth', 'toothbrush', 'toothpaste', 'tornado', 'tractor', 'tree', 'truck', 't-shirt', 'umbrella', 'underwear', 'vase', 'violin', 'watermelon', 'whale', 'wine_glass', 'zebra']

# from ..build import DATASET_REGISTRY
# from ..base_dataset import Datum, DatasetBase


@DATASET_REGISTRY.register()
class DomainNetDF(DatasetBase):
    """DomainNet.

    Statistics:
        - 6 distinct domains: Clipart, Infograph, Painting, Quickdraw,
        Real, Sketch.
        - Around 0.6M images.
        - 345 categories.
        - URL: http://ai.bu.edu/M3SDA/.

    Special note: the t-shirt class (327) is missing in painting_train.txt.

    Reference:
        - Peng et al. Moment Matching for Multi-Source Domain
        Adaptation. ICCV 2019.
    """

    dataset_dir = "DomainNet"
    domains = [
        "clipart", "infograph", "painting", "quickdraw", "real", "sketch"
    ]

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.split_dir = self.dataset_dir

        # self.check_input_domains(
        #     cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        # )
        train_domains = [
            "clipart", "infograph", "painting", "quickdraw", "real", "sketch"
        ]

        # train_domains = ["clipart", "painting", "real", "sketch"]
        # test_domains = ["clipart", "painting", "real", "sketch"]
        test_domains = [
            "clipart", "infograph", "painting", "quickdraw", "real", "sketch"
        ]
        train_x = self._read_data(train_domains, split="train")
        train_u = self._read_data(train_domains, split="train")
        val = self._read_data(test_domains, split="test")
        test = self._read_data(test_domains, split="test")

        num_shots = cfg.DATASET.NUM_SHOTS  # 使用する数ショット数を設定
        train_x = self.generate_fewshot_dataset(train_x, num_shots=num_shots, repeat=True, seed=cfg.SEED)

        super().__init__(train_x=train_x, train_u=train_u, val=val, test=test)
    
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

    def _read_data(self, input_domains, split="train"):
        items = []

        for domain, dname in enumerate(input_domains):
            filename = dname + "_" + split + ".txt"
            split_file = osp.join(self.split_dir, filename)

            with open(split_file, "r") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    impath, label = line.split(" ")
                    classname = impath.split("/")[1]
                    impath = osp.join(self.dataset_dir, impath)
                    label = int(label)
                    item = Datum(
                        impath=impath,
                        label=label,
                        domain=domain,
                        classname=classname
                    )
                    items.append(item)

        return items

# class DomainNetDF(DatasetBase):
#     dataset_dir = "domainnet"
    
#     def __init__(self, cfg):
#         root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
#         self.dataset_dir = os.path.join(root, self.dataset_dir)
#         self.image_dir = self.dataset_dir# os.path.join(self.dataset_dir, "101_ObjectCategories")
#         self.split_path = os.path.join(self.dataset_dir, "data_split_DF.json")
#         train, test = read_split(filepath=self.split_path, path_prefix=self.image_dir, use_domain=DOMAIN_NAMES) #FIXME
#         super().__init__(train_x=train, test=test)

# def read_split(filepath, path_prefix, use_domain:List[str]):
#     def _convert(items):
#         out = []
#         for impath, label, domain, classname in items:
#             impath = os.path.join(path_prefix, impath)
#             item = Datum(impath=impath, label=int(label), domain=int(domain), classname=str(classname))
#             out.append(item)
#         return out
#     print(f"Reading split from {filepath}")
#     split = read_json(filepath)
#     train, test = [], []
#     for domain in use_domain:
#         train += _convert(split["train"][str(domain)])
#         test += _convert(split["test"][str(domain)])
    
#     return train, test
    
