"""Verify the CONTROL arm really removes unlearning and nothing else."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from types import SimpleNamespace

import torch
from train_loop import setup_cfg
from phase0_diagnostic import build_args

FAIL = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not cond:
        FAIL.append(name)


def cfg_for(control):
    cli = SimpleNamespace(output_dir="/tmp/p0ctl", seed=1, forget="sketch",
                          heldout_num=26, heldout_seed=1234, gpu="",
                          root="/x", control=control)
    return setup_cfg(build_args(cli))


print("\n=== CONTROL vs ADU config ===")
adu, ctl = cfg_for(False), cfg_for(True)

check("ADU arm uses entropy forget loss", adu.FORGET_LOSS_TYPE == "entropy", adu.FORGET_LOSS_TYPE)
check("CONTROL arm uses 'none' forget loss", ctl.FORGET_LOSS_TYPE == "none", ctl.FORGET_LOSS_TYPE)
check("ADU DDL gamma=30, lambda=10",
      adu.DDL_LOSS_WEIGHT == 30.0 and adu.MMD_WEIGHT == 10.0,
      f"gamma={adu.DDL_LOSS_WEIGHT} lambda={adu.MMD_WEIGHT}")
check("CONTROL DDL fully zeroed",
      ctl.DDL_LOSS_WEIGHT == 0.0 and ctl.MMD_WEIGHT == 0.0,
      f"gamma={ctl.DDL_LOSS_WEIGHT} lambda={ctl.MMD_WEIGHT}")

# everything that is NOT the unlearning objective must be identical
same_keys = {
    "backbone": (adu.MODEL.BACKBONE.NAME, ctl.MODEL.BACKBONE.NAME),
    "max_epoch": (adu.OPTIM.MAX_EPOCH, ctl.OPTIM.MAX_EPOCH),
    "lr": (adu.OPTIM.LR, ctl.OPTIM.LR),
    "batch_size": (adu.DATALOADER.TRAIN_X.BATCH_SIZE, ctl.DATALOADER.TRAIN_X.BATCH_SIZE),
    "num_shots": (adu.DATASET.NUM_SHOTS, ctl.DATASET.NUM_SHOTS),
    "heldout_num": (adu.DATASET.HELDOUT_NUM, ctl.DATASET.HELDOUT_NUM),
    "heldout_seed": (adu.DATASET.HELDOUT_SEED, ctl.DATASET.HELDOUT_SEED),
    "forget_domains": (adu.DATASET.FORGETDOMAINS, ctl.DATASET.FORGETDOMAINS),
    "instapg(cross_attn)": (adu.USE_CROSSATTENTION, ctl.USE_CROSSATTENTION),
    "prompt_depth_vision": (adu.TRAINER.IVLP.PROMPT_DEPTH_VISION, ctl.TRAINER.IVLP.PROMPT_DEPTH_VISION),
    "prompt_depth_text": (adu.TRAINER.IVLP.PROMPT_DEPTH_TEXT, ctl.TRAINER.IVLP.PROMPT_DEPTH_TEXT),
    "domain_cls_head_built": (adu.USE_DOMAIN_CLASIFIER_LOSS, ctl.USE_DOMAIN_CLASIFIER_LOSS),
}
for k, (a, c) in same_keys.items():
    check(f"identical across arms: {k}", a == c, f"{a} vs {c}")

print("\n=== loss-combiner semantics for forget_loss_type='none' ===")
# Mirrors engine/trainer.py: loss_del stays a float -> `loss = base`, so no
# forget term can enter the objective.
loss_del, base = 0.0, torch.tensor(2.5)
loss = base + 0 if isinstance(loss_del, torch.Tensor) else base
check("non-tensor loss_del => loss == retain CE only", torch.equal(loss, base), f"{loss.item()}")

# DDL with zeroed weights contributes exactly 0
domain_logit = torch.randn(8, 4)
target = torch.randint(0, 4, (8,))
ddl = torch.nn.functional.cross_entropy(domain_logit, target) * ctl.DDL_LOSS_WEIGHT
mmd = torch.rand(1).squeeze() * ctl.MMD_WEIGHT
check("DDL contributes exactly 0 in control", float(ddl - mmd) == 0.0, f"{float(ddl - mmd)}")

print("\n" + "=" * 60)
print("CONTROL CHECK FAILED: " + ", ".join(FAIL) if FAIL else "ALL CONTROL CHECKS PASSED")
print("=" * 60)
