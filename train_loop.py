import argparse
import torch
import csv

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

# custom
import datasets.domainnet_df
import datasets.office_home_df
import datasets.domainnet_mini_df

import trainers.independent_VLAdapter_Prompt

from typing import List, Tuple, Dict
import itertools
from datetime import datetime
import os
import os.path as osp
import copy

def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head

    if args.num_shots:
        cfg.DATASET.NUM_SHOTS = args.num_shots


def extend_cfg(cfg, args):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    if len(args.forget_domains) > 0:
        if "none" in args.forget_domains:
            cfg.DATASET.FORGETDOMAINS = []
        else :
            cfg.DATASET.FORGETDOMAINS = args.forget_domains
    else :
        cfg.DATASET.FORGETDOMAINS = args.forget_domains
        # print(cfg.DATASET.FORGETDOMAINS)
    cfg.EVAL_ONLY = args.eval_only
    cfg.DATASET.SEED = args.seed
    cfg.LMD_DOMAIN_LOSS = args.lmd_domain_loss
    cfg.USE_DOMAIN_CLASIFIER_LOSS = args.use_domain_cls_loss
    cfg.DOMAIN_CLASS_DIVIDED = args.domain_class_divided
    cfg.IS_DOMAIN_DIVIDED = args.is_domain_divided
    cfg.CSV_FILE_PATH = args.csv_file_path
    cfg.MMD_WEIGHT = args.mmd_weight 
    cfg.SOFT_LABEL_UPDATE_EPOCH = 1
    cfg.USE_SOFT_DOMAIN_LABEL = False
    cfg.PREPROCESS_SOFT_LABEL = "Total" # Total or Class
    cfg.USE_KLDIV_PENALTY = None
    cfg.ONLY_KLDIV_FOR_PRV = False
    cfg.ADD_LINEAR = False
    cfg.USE_CLASSTOKEN = False
    cfg.USE_CROSSATTENTION = True
    cfg.USE_VISION_ADAPTER = False
    cfg.USE_TEXT_ADAPTER = False
    cfg.INDEPENDENT_CROSS_ATTENTION = False
    cfg.INDEPENDENT_LEARNABLE_VISION = True
    cfg.INSERT_LAYER_ATTN = 9
    cfg.USE_ORTHOGONAL_LOSS = False
    cfg.DDL_LOSS_WEIGHT = args.domainloss_weight # これでDomain Classifierの損失の重みをとるようにしてね
    cfg.TRAINER.IVLP_VL_Adapter_Local = CN()
    cfg.TRAINER.IVLP_VL_Adapter_Local.BLOCK_SHUFFLE_SELECT_NON_EXPERT = False
    cfg.TRAINER.ClipFit_DF = CN()
    cfg.TRAINER.ClipFit_DF.USE_KD = True
    cfg.TRAINER.IVLP_VLADAPTER_LOCAL_SELECTPATCH = CN()
    cfg.TRAINER.IVLP_VLADAPTER_LOCAL_SELECTPATCH.TOPK = 190
    cfg.TRAINER.IVLP_VLADAPTER_LOCAL_SELECTPATCH.ONLY_MASKED = False
    cfg.TRAINER.IVLP_VLADAPTER_LOCAL_SELECTPATCH.SELECT_METHOD = "block_shuffle_distill"
    cfg.TRAINER.IVLP_VLADAPTER_LOCAL_SELECTPATCH.SELECT_LAYER = 9
    cfg.TRAINER.COOP_W_ADAPTER = CN()
    cfg.TRAINER.DOMAINCLS = CN()
    cfg.TRAINER.DOMAINCLS.PREC = "fp16" 
    cfg.TRAINER.COOP_W_DH = CN()
    cfg.TRAINER.COOP_W_DH.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COOP_W_DH.CSC = False  # class-specific context
    cfg.TRAINER.COOP_W_DH.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP_W_DH.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COOP_W_DH.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.CLIP_Adapter = CN()
    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp16"  # fp16, fp32, amp

    # Config for MaPLe
    cfg.TRAINER.MAPLE = CN()
    cfg.TRAINER.MAPLE.N_CTX = 2  # number of context vectors
    cfg.TRAINER.MAPLE.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.MAPLE.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.MAPLE.PROMPT_DEPTH = 9  # Max 12, minimum 0, for 1 it will act as shallow MaPLe (J=1)
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    # Config for PromptSRC
    cfg.TRAINER.PROMPTSRC = CN()
    cfg.TRAINER.PROMPTSRC.N_CTX_VISION = 4  # number of context vectors at the vision branch
    cfg.TRAINER.PROMPTSRC.N_CTX_TEXT = 4  # number of context vectors at the language branch
    cfg.TRAINER.PROMPTSRC.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.PROMPTSRC.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.PROMPTSRC.PROMPT_DEPTH_VISION = 9  # Max 12, minimum 0, for 0 it will be using shallow IVLP prompting (J=1)
    cfg.TRAINER.PROMPTSRC.PROMPT_DEPTH_TEXT = 9  # Max 12, minimum 0, for 0 it will be using shallow IVLP prompting (J=1)
    cfg.TRAINER.PROMPTSRC.TEXT_LOSS_WEIGHT = 25
    cfg.TRAINER.PROMPTSRC.IMAGE_LOSS_WEIGHT = 10
    cfg.TRAINER.PROMPTSRC.GPA_MEAN = 15
    cfg.TRAINER.PROMPTSRC.GPA_STD = 1
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    # Config for independent Vision Language prompting (independent-vlp)
    cfg.TRAINER.IVLP = CN()
    cfg.TRAINER.IVLP.N_CTX_VISION = 2  # number of context vectors at the vision branch
    cfg.TRAINER.IVLP.N_CTX_TEXT = 2  # number of context vectors at the language branch
    cfg.TRAINER.IVLP.CTX_INIT = "a photo of a"  # initialization words (only for language prompts)
    cfg.TRAINER.IVLP.PREC = "fp16"  # fp16, fp32, amp

    # If both variables below are set to 0, 0, will the config will degenerate to COOP model
    cfg.TRAINER.IVLP.PROMPT_DEPTH_VISION = 9  # Max 12, minimum 0, for 0 it will act as shallow IVLP prompting (J=1)
    cfg.TRAINER.IVLP.PROMPT_DEPTH_TEXT = 9  # Max 12, minimum 0, for 0 it will act as shallow IVLP prompting(J=1)
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    # Config for only vision side prompting MaPLEの実装から流用
    cfg.TRAINER.VPT = CN()
    cfg.TRAINER.VPT.N_CTX_VISION = 8  # number of context vectors at the vision branch
    cfg.TRAINER.VPT.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.VPT.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.VPT.PROMPT_DEPTH_VISION = 1  # if set to 1, will represent shallow vision prompting only
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new
    cfg.TRAINER.CoOpDomainSpecific = CN()
    # cfg.TRAINER.CLIP_Adapter = CN()

def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg, args)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)

    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    if not args.no_train:
        results = trainer.train_loop()
        return results
    
def get_loop_prepare(datasetname: str)->Tuple[List[str], Dict]:
    print(datasetname)
    if datasetname == "office_home_df":
        domain_list = ["art", "clipart", "product", "real_world"]

    elif datasetname == "domainnet_df":
        domain_list = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]

    elif datasetname == "domainnet_mini_df":
        domain_list = ["clipart", "painting", "real", "sketch"]
    else :
        assert False, "Dataset name should be office_home_df or domainnet_mini_df or domainnet_df"
    
    base_dict = {
            "A" : [],
            "F" : [],
            "H" : []
        }

    power_set = [
        list(subset) for i in range(1, len(domain_list)) \
            for subset in itertools.combinations(domain_list, i)
    ]

    res_dict = {}
    for i in range(1, len(domain_list)):
        key_i = f"forgetdomain_{i}"
        res_dict[key_i] = copy.deepcopy(base_dict)

    return power_set, res_dict

def create_csv_file(filename:str, forget_domain_num: int):
    data = [
        ["EXPNAME", ""],
        ["", "DATE"]
    ]
    for idx in range(forget_domain_num):
        data[0].extend(["", f"Forgetdomain{idx+1}", ""])
        data[1].extend(["H", "A", "F"])
    # data[1].extend(["ave", "std"])
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows(data)
    print(f"initialize csv file: {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument( "--root", type=str, default="", help="path to dataset")
    parser.add_argument( "--output-dir", type=str, default="", help="output directory")
    parser.add_argument( "--resume", type=str, default="", help="checkpoint directory (from which the training resumes)",)
    parser.add_argument( "--seed", type=int, default=-1, help="only positive value enables a fixed seed")
    parser.add_argument( "--source-domains", type=str, nargs="+", help="source domains for DA/DG")
    parser.add_argument( "--target-domains", type=str, nargs="+", help="target domains for DA/DG")
    parser.add_argument( "--transforms", type=str, nargs="+", help="data augmentation methods")
    parser.add_argument( "--config-file", type=str, default="", help="path to config file")
    parser.add_argument( "--dataset-config-file", type=str, default="", help="path to config file for dataset setup",)
    parser.add_argument( "--trainer", type=str, default="", help="name of trainer")
    parser.add_argument( "--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument( "--head", type=str, default="", help="name of head")
    parser.add_argument( "--eval-only", action="store_true", help="evaluation only")
    parser.add_argument( "--model-dir", type=str, default="", help="load model from this directory for eval-only mode",)
    parser.add_argument( "--load-epoch", type=int, help="load model weights at this epoch for evaluation")
    parser.add_argument( "--no-train", action="store_true", help="do not call trainer.train()")
    parser.add_argument( "--num_shots", type=int, default=-1)
    parser.add_argument( "--forget_domains", default=[], nargs="*", help="input forget domains like '--forget_domains domain1 domain2 ..' ")
    parser.add_argument( "--block_shuffle_selection", action="store_true", help="default is False")
    parser.add_argument( "--block_shuffle_selection_nonexp", action="store_true", help="default is False")
    parser.add_argument( "--topk", default=100, type=int, help="select local feat topk ")
    parser.add_argument( "--domain_class_divided", action="store_true", help="default is False")
    parser.add_argument( "--lmd_domain_loss", type=float, default=1.0)
    parser.add_argument( "--use_domain_cls_loss", action="store_true", help="default is False")
    parser.add_argument( "--is_domain_divided", action="store_true", help="defult is False")
    parser.add_argument( "--csv_file_path", type=str, default="default.csv")
    parser.add_argument( "--dataset_name", type=str, default="")
    parser.add_argument( "--experiment_name", type=str, default="exp")
    parser.add_argument( "--sub_experiment_name", type=str, default="subexp")
    parser.add_argument( "--domainloss_weight", type=float, default=0.0)
    parser.add_argument( "--mmd_weight", type=float, default=0.0)
    parser.add_argument( "opts", default=None, nargs=argparse.REMAINDER, help="modify config options using the command-line",)

    args = parser.parse_args()
    
    forget_domain_lists, base_dict = get_loop_prepare(args.dataset_name)

    datasetname = args.dataset_name

    base_output_dir = args.output_dir + "/"

    seed_list = [1,2,3]

    # output csv file path
    exp_csv_filedir = args.output_dir + "/"
    exp_csv_filepath = exp_csv_filedir + f"/outputs.csv"

    if not osp.exists(exp_csv_filedir):
        os.makedirs(exp_csv_filedir)

    if not osp.exists(exp_csv_filepath):
        create_csv_file(exp_csv_filepath, len(base_dict))

    results_dict = {}
    for seed in seed_list:

        exp_csv_filepath_seedwise = exp_csv_filedir + "/" + f"results_seed{seed}.csv"
        if not osp.exists(exp_csv_filepath_seedwise):
            create_csv_file(exp_csv_filepath_seedwise, len(base_dict))

        results_dict[f"seed{seed}"] = copy.deepcopy(base_dict)

        for forget_domain_list in forget_domain_lists:
            args.forget_domains = forget_domain_list
            args.seed = seed

            now = datetime.now()
            today = now.strftime("%Y%m%d_%H%M%S")
            forget_domain_str = "-".join(forget_domain_list)
            args.output_dir = base_output_dir + f"/seed{seed}/ForgetDomain{len(forget_domain_list)}/{forget_domain_str}/{today}" 
            args.csv_file_path = base_output_dir + f"/seed{seed}/ForgetDomain{len(forget_domain_list)}/results.csv"

            results = main(args)

            results_dict[f"seed{seed}"][f"forgetdomain_{len(forget_domain_list)}"]["A"].append(results["A"])
            results_dict[f"seed{seed}"][f"forgetdomain_{len(forget_domain_list)}"]["F"].append(results["F"])
            results_dict[f"seed{seed}"][f"forgetdomain_{len(forget_domain_list)}"]["H"].append(results["H"])
        
        now = datetime.now()
        today = now.strftime("%Y%m%d_%H%M%S")
        data_seed = [today,]

        for idx in range(len(results_dict[f"seed{seed}"])):
            results_dict[f"seed{seed}"][f"forgetdomain_{idx+1}"]["A"] = sum(results_dict[f"seed{seed}"][f"forgetdomain_{idx+1}"]["A"]) / len(results_dict[f"seed{seed}"][f"forgetdomain_{idx+1}"]["A"])
            results_dict[f"seed{seed}"][f"forgetdomain_{idx+1}"]["F"] = sum(results_dict[f"seed{seed}"][f"forgetdomain_{idx+1}"]["F"]) / len(results_dict[f"seed{seed}"][f"forgetdomain_{idx+1}"]["F"])
            results_dict[f"seed{seed}"][f"forgetdomain_{idx+1}"]["H"] = sum(results_dict[f"seed{seed}"][f"forgetdomain_{idx+1}"]["H"]) / len(results_dict[f"seed{seed}"][f"forgetdomain_{idx+1}"]["H"])
            data_seed.extend(
                [
                    results_dict[f"seed{seed}"][f"forgetdomain_{idx+1}"]["H"],
                    results_dict[f"seed{seed}"][f"forgetdomain_{idx+1}"]["A"],
                    results_dict[f"seed{seed}"][f"forgetdomain_{idx+1}"]["F"]
                ]
            )  
        
        with open(exp_csv_filepath_seedwise, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(data_seed)

    tot_res = len(results_dict) # seed num
    tot_num_fd = len(base_dict) # forget domain num 

    fin_res = copy.deepcopy(base_dict)
    for num_fd in range(tot_num_fd):
        for s in seed_list:
            fin_res[f"forgetdomain_{num_fd + 1}"]["H"].append(results_dict[f"seed{s}"][f"forgetdomain_{num_fd+1}"]["H"])
            fin_res[f"forgetdomain_{num_fd + 1}"]["A"].append(results_dict[f"seed{s}"][f"forgetdomain_{num_fd+1}"]["A"])
            fin_res[f"forgetdomain_{num_fd + 1}"]["F"].append(results_dict[f"seed{s}"][f"forgetdomain_{num_fd+1}"]["F"])
        
    now = datetime.now()
    today = now.strftime("%Y%m%d_%H%M%S")
    data_tot = [today,]
    for num_fd in range(tot_num_fd):
        fin_res[f"forgetdomain_{num_fd+1}"]["H"] = sum(fin_res[f"forgetdomain_{num_fd+1}"]["H"]) / len (fin_res[f"forgetdomain_{num_fd+1}"]["H"])
        fin_res[f"forgetdomain_{num_fd+1}"]["A"] = sum(fin_res[f"forgetdomain_{num_fd+1}"]["A"]) / len (fin_res[f"forgetdomain_{num_fd+1}"]["A"])
        fin_res[f"forgetdomain_{num_fd+1}"]["F"] = sum(fin_res[f"forgetdomain_{num_fd+1}"]["F"]) / len (fin_res[f"forgetdomain_{num_fd+1}"]["F"])
        data_tot.extend(
            [
                fin_res[f"forgetdomain_{num_fd+1}"]["H"],
                fin_res[f"forgetdomain_{num_fd+1}"]["A"],
                fin_res[f"forgetdomain_{num_fd+1}"]["F"]
            ]
        ) 

    with open(exp_csv_filepath, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(data_tot)

    

