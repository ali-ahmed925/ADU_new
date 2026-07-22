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
import datasets.domainnet_mini_paper_df

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
    cfg.DATASET.FORGETCLASSES = getattr(args, "forget_classes", [])
    cfg.FORGET_LOSS_TYPE = getattr(args, "forget_loss_type", "entropy")
    cfg.NO_RETAIN_LOSS = getattr(args, "no_retain_loss", False)
    cfg.FORGET_WEIGHT = getattr(args, "forget_weight", 1.0)
    cfg.FLAT_WEIGHT = getattr(args, "flat_weight", 1.0)
    cfg.SUPPRESS_CAP = getattr(args, "suppress_cap", 6.0)
    cfg.MARG_WEIGHT = getattr(args, "marg_weight", 1.0)
    cfg.FORGET_POOL_SIZE = getattr(args, "forget_pool_size", 0)
    cfg.FORGET_CHUNK = getattr(args, "forget_chunk", 0)
    cfg.EXCLUDE_FORGET_CLASS_FROM_RETAIN = getattr(args, "exclude_forget_class_from_retain", False)
    cfg.EVAL_ONLY = args.eval_only
    cfg.DATASET.SEED = args.seed
    cfg.USE_DOMAIN_CLASIFIER_LOSS = args.use_domain_cls_loss
    cfg.IS_DOMAIN_DIVIDED = args.is_domain_divided
    cfg.CSV_FILE_PATH = args.csv_file_path
    cfg.MMD_WEIGHT = args.mmd_weight 
    cfg.USE_CLASSTOKEN = False
    cfg.USE_CROSSATTENTION = True
    cfg.USE_VISION_ADAPTER = False
    cfg.USE_TEXT_ADAPTER = False
    cfg.INDEPENDENT_CROSS_ATTENTION = False
    cfg.INDEPENDENT_LEARNABLE_VISION = True
    cfg.INSERT_LAYER_ATTN = 9
    cfg.DDL_LOSS_WEIGHT = args.domainloss_weight
    # Subspace-constrained DDL: feed the domain classifier the component of the
    # image feature orthogonal to the frozen zero-shot class subspace.
    cfg.SUBSPACE_DDL = getattr(args, "subspace_ddl", False)

    # Config for independent Vision Language prompting (independent-vlp)
    cfg.TRAINER.IVLP = CN()
    cfg.TRAINER.IVLP.N_CTX_VISION = 2  # number of context vectors at the vision branch
    cfg.TRAINER.IVLP.N_CTX_TEXT = 2  # number of context vectors at the language branch
    cfg.TRAINER.IVLP.CTX_INIT = "a photo of a"  # initialization words (only for language prompts)
    cfg.TRAINER.IVLP.PREC = "fp16"  # fp16, fp32, amp

    cfg.TRAINER.IVLP.PROMPT_DEPTH_VISION = 9  
    cfg.TRAINER.IVLP.PROMPT_DEPTH_TEXT = 9  
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"

    # Phase-0 open-vocabulary diagnostic: hold out N classes from TRAIN only
    # (test keeps all 126 classes / text anchors). Default 0 => stock ADU.
    cfg.DATASET.HELDOUT_NUM = getattr(args, "heldout_num", 0)
    cfg.DATASET.HELDOUT_SEED = getattr(args, "heldout_seed", 1234)

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
    
def get_loop_prepare(datasetname: str, run_forget_domains: List[str] = None)->Tuple[List[str], Dict]:
    print(datasetname)
    if datasetname == "office_home_df":
        domain_list = ["art", "clipart", "product", "real_world"]

    elif datasetname == "domainnet_df":
        domain_list = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]

    elif datasetname in ("domainnet_mini_df", "domainnet_mini_paper_df"):
        domain_list = ["clipart", "painting", "real", "sketch"]
    else :
        assert False, "Dataset name should be office_home_df or domainnet_mini_df or domainnet_mini_paper_df or domainnet_df"

    base_dict = {
            "A" : [],
            "F" : [],
            "H" : []
        }

    if run_forget_domains:
        # Only run the specified exact combination (e.g. ["sketch"])
        power_set = [sorted(run_forget_domains, key=lambda d: domain_list.index(d))]
    else:
        power_set = [
            list(subset) for i in range(1, len(domain_list)) \
                for subset in itertools.combinations(domain_list, i)
        ]

    res_dict = {}
    sizes = sorted({len(subset) for subset in power_set})
    for i in sizes:
        res_dict[f"forgetdomain_{i}"] = copy.deepcopy(base_dict)

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
    parser.add_argument( "--forget_classes", default=[], nargs="*", help="class names to forget within the forget domains, e.g. '--forget_classes tiger lion'")
    parser.add_argument( "--forget_loss_type", type=str, default="entropy", choices=["entropy", "neggrad", "flat", "suppress_flat", "suppress_entropy", "suppress_marg", "none"],
        help="forget objective: 'none' = CONTROL, no unlearning at all (pair with "
             "--domainloss_weight 0 --mmd_weight 0 for plain prompt tuning), "
             "'entropy' = entropy maximization (ADU-style), 'neggrad' = gradient ascent on forget-set CE, "
             "'flat' = minimize variance of scaled logits (uniform-by-construction, leak-free target), "
             "'suppress_flat' (P2) = suppress target logit (NegGrad-strength) + flatten non-target logits (variance-min, feature-level), "
             "'suppress_entropy' (P2b) = suppress target logit + maximize non-target entropy (output-level, avoids feature funnelling), "
             "'suppress_marg' (P2c) = suppress + per-image entropy + BATCH-MARGINAL entropy (breaks the across-image funnel; forwards all forget images together each step)")
    parser.add_argument( "--marg_weight", type=float, default=1.0,
        help="P2c only: weight on the batch-marginal entropy term in 'suppress_marg'")
    parser.add_argument( "--forget_pool_size", type=int, default=0,
        help="P2c only: number of forget images for the dedicated forget forward (0=use the 8-shot set). "
             ">0 loads that many forget (class,domain) images from the split so the batch-marginal is non-degenerate; retain stays 8-shot")
    parser.add_argument( "--forget_chunk", type=int, default=0,
        help="P2c only: how many pool images to forward PER STEP (0 = all). Decouples pool "
             "diversity from GPU memory; a random chunk each step also gives an unbiased marginal estimate")
    parser.add_argument( "--no_retain_loss", action="store_true",
        help="drop the retain CE term (pure NegGrad: ascent on forget set only)")
    parser.add_argument( "--forget_weight", type=float, default=1.0,
        help="weight (lambda_f) on the whole forget term ('flat' / 'suppress_flat')")
    parser.add_argument( "--flat_weight", type=float, default=1.0,
        help="P2 only: relative weight of the flatten (anti-leak) term vs the suppression term in 'suppress_flat'")
    parser.add_argument( "--suppress_cap", type=float, default=6.0,
        help="P2 only: cap on per-image suppression CE (prevents unbounded push-down / NegGrad blow-up). "
             "ln(126)=4.84 is uniform; 6.0 => target prob ~0.0025, below uniform")
    parser.add_argument( "--exclude_forget_class_from_retain", action="store_true",
        help="Lever P1: remove the forget class (ALL domains) from the retain CE, "
             "so nothing reinforces it while forgetting propagates. Default OFF (v1).")
    parser.add_argument( "--domain_class_divided", action="store_true", help="default is False")
    parser.add_argument( "--lmd_domain_loss", type=float, default=1.0)
    parser.add_argument( "--use_domain_cls_loss", action="store_true", help="default is False")
    parser.add_argument( "--is_domain_divided", action="store_true", help="defult is False")
    parser.add_argument( "--csv_file_path", type=str, default="default.csv")
    parser.add_argument( "--dataset_name", type=str, default="")
    parser.add_argument( "--domainloss_weight", type=float, default=0.0)
    parser.add_argument( "--mmd_weight", type=float, default=0.0)
    parser.add_argument( "--subspace_ddl", action="store_true",
        help="Subspace-constrained DDL: feed the domain classifier the component "
             "of the image feature orthogonal to the frozen zero-shot class "
             "subspace, so domain separation cannot consume the directions the "
             "zero-shot classifier reads.")
    parser.add_argument( "--heldout_num", type=int, default=0,
        help="Phase-0 diagnostic: number of classes to hold out of TRAINING only "
             "(test keeps all classes/text anchors). 0 = stock ADU.")
    parser.add_argument( "--heldout_seed", type=int, default=1234,
        help="Seed selecting which classes are held out (fixed across model seeds).")
    parser.add_argument( "--run_forget_domains", default=[], nargs="*",
        help="Run only this specific forget-domain combination (e.g. --run_forget_domains sketch). "
             "If empty, runs the full power-set of all domains (paper default).")
    parser.add_argument( "--seeds", default=[1, 2, 3], nargs="+", type=int,
        help="Which seeds to run (default: 1 2 3). E.g. --seeds 2 3 to skip seed 1.")
    parser.add_argument( "opts", default=None, nargs=argparse.REMAINDER, help="modify config options using the command-line",)

    args = parser.parse_args()

    run_filter = args.run_forget_domains if args.run_forget_domains else None
    forget_domain_lists, base_dict = get_loop_prepare(args.dataset_name, run_forget_domains=run_filter)

    datasetname = args.dataset_name

    base_output_dir = args.output_dir + "/"

    seed_list = args.seeds

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

        for key in results_dict[f"seed{seed}"]:
            for metric in ("A", "F", "H"):
                vals = results_dict[f"seed{seed}"][key][metric]
                results_dict[f"seed{seed}"][key][metric] = sum(vals) / len(vals)
            data_seed.extend(
                [
                    results_dict[f"seed{seed}"][key]["H"],
                    results_dict[f"seed{seed}"][key]["A"],
                    results_dict[f"seed{seed}"][key]["F"]
                ]
            )
        
        with open(exp_csv_filepath_seedwise, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(data_seed)

    fin_res = copy.deepcopy(base_dict)
    for key in fin_res:
        for s in seed_list:
            fin_res[key]["H"].append(results_dict[f"seed{s}"][key]["H"])
            fin_res[key]["A"].append(results_dict[f"seed{s}"][key]["A"])
            fin_res[key]["F"].append(results_dict[f"seed{s}"][key]["F"])

    now = datetime.now()
    today = now.strftime("%Y%m%d_%H%M%S")
    data_tot = [today,]
    for key in fin_res:
        for metric in ("H", "A", "F"):
            fin_res[key][metric] = sum(fin_res[key][metric]) / len(fin_res[key][metric])
        data_tot.extend(
            [
                fin_res[key]["H"],
                fin_res[key]["A"],
                fin_res[key]["F"]
            ]
        )

    with open(exp_csv_filepath, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(data_tot)

    

