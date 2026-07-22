"""Verify Tier 1 (subspace-constrained DDL): shapes, gradients, one-time SVD."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from types import SimpleNamespace

import torch
from train_loop import setup_cfg
from phase0_diagnostic import build_args
import trainers.independent_VLAdapter_Prompt as T

FAIL = []
def chk(n, c, d=""):
    print(f"  [{'PASS' if c else 'FAIL'}] {n}" + (f"  -- {d}" if d else ""))
    if not c: FAIL.append(n)


def cfg(subspace):
    cli = SimpleNamespace(output_dir="/tmp/t1", seed=1, forget="real", heldout_num=26,
                          heldout_seed=1234, gpu="", root="/x", arm="adu",
                          control=False, subspace_ddl=subspace)
    return setup_cfg(build_args(cli))


print("\n=== config plumbing ===")
chk("SUBSPACE_DDL False by default", cfg(False).SUBSPACE_DDL is False)
chk("SUBSPACE_DDL True when flagged", cfg(True).SUBSPACE_DDL is True)
c = cfg(True)
chk("DDL weights untouched by the flag",
    c.DDL_LOSS_WEIGHT == 30.0 and c.MMD_WEIGHT == 10.0,
    f"gamma={c.DDL_LOSS_WEIGHT} lambda={c.MMD_WEIGHT}")
chk("forget loss untouched by the flag", c.FORGET_LOSS_TYPE == "entropy")

print("\n=== projection math (synthetic basis) ===")
d, r, n = 512, 126, 32
g = torch.Generator().manual_seed(0)
B = torch.linalg.qr(torch.randn(d, r, generator=g))[0].T.contiguous()   # (r,d) orthonormal
chk("basis is orthonormal (B B^T = I)",
    torch.allclose(B @ B.T, torch.eye(r), atol=1e-5))

z = torch.randn(n, d, generator=g)
z = z / z.norm(dim=1, keepdim=True)
z_perp = z - (z @ B.t()) @ B
z_par = z - z_perp
chk("z_perp shape == z shape (projection, not reduction)", z_perp.shape == z.shape,
    str(tuple(z_perp.shape)))
chk("z_perp orthogonal to span(B)", float((z_perp @ B.t()).abs().max()) < 1e-4,
    f"{float((z_perp @ B.t()).abs().max()):.2e}")
chk("z_par + z_perp == z", torch.allclose(z_par + z_perp, z, atol=1e-5))
chk("idempotent (projecting twice == once)",
    torch.allclose(z_perp - (z_perp @ B.t()) @ B, z_perp, atol=1e-5))
chk("energy split sums to 1",
    abs(float((z_par**2).sum(1).mean() + (z_perp**2).sum(1).mean()) - 1.0) < 1e-4)

print("\n=== gradient flow ===")
zg = z.clone().requires_grad_(True)
Bb = B.clone()                       # buffer: no grad
out = (zg - (zg @ Bb.t()) @ Bb).sum()
out.backward()
chk("gradient reaches image features", zg.grad is not None and zg.grad.abs().sum() > 0)
chk("basis carries no gradient", Bb.grad is None)
# gradient of sum(z_perp) wrt z is (I - B^T B) applied to ones -> must be in perp space
gp = zg.grad
chk("gradient lies in the orthogonal complement",
    float((gp @ B.t()).abs().max()) < 1e-3, f"{float((gp @ B.t()).abs().max()):.2e}")

print("\n=== CustomCLIP wiring (no CLIP download needed) ===")
class Dummy(torch.nn.Module):
    """Mirrors the real CustomCLIP wiring for the subspace bits only."""
    def __init__(self):
        super().__init__()
        self.subspace_ddl = True
        self.dtype = torch.float32
        self.domain_classifier = torch.nn.Linear(d, 4)
        self.register_buffer("cls_basis", None, persistent=False)
    set_class_subspace_basis = T.CustomCLIP.set_class_subspace_basis

m = Dummy()
chk("no basis installed initially", m.cls_basis is None)
m.set_class_subspace_basis(B)
chk("basis registered as a buffer", "cls_basis" in dict(m.named_buffers()))
chk("basis NOT in state_dict (persistent=False)", "cls_basis" not in m.state_dict())
chk("basis requires_grad == False", m.cls_basis.requires_grad is False)
chk("basis dtype matches model dtype", m.cls_basis.dtype == m.dtype)

# the exact expression used in CustomCLIP.forward
feat = torch.randn(n, d, generator=g)
b = m.cls_basis
domain_input = feat - (feat @ b.t()) @ b
chk("domain classifier accepts projected input",
    m.domain_classifier(domain_input).shape == (n, 4),
    str(tuple(m.domain_classifier(domain_input).shape)))
chk("projected input differs from raw", not torch.allclose(domain_input, feat))

print("\n=== SVD computed once ===")
src = open("trainers/independent_VLAdapter_Prompt.py").read()
chk("svd appears only inside compute_frozen_class_subspace",
    src.count("linalg.svd") == 1, f"{src.count('linalg.svd')} occurrences")
fwd = src.split("def forward(self, image, label=None):")[1].split("class Adapter")[0]
chk("no svd in forward()", "svd" not in fwd)
chk("basis installed in build_model, not per step",
    "set_class_subspace_basis" in src.split("def build_model")[1].split("name_to_update")[0])

print("\n=== trainer registry intact ===")
# Regression guard: inserting a module-level def directly beneath
# @TRAINER_REGISTRY.register() silently registers the FUNCTION instead of the
# trainer class, and build_trainer then fails with "expected to belong to [...]".
from dassl.engine import TRAINER_REGISTRY
names = TRAINER_REGISTRY.registered_names()
chk("IVLP_VL_Adapter_Prompt is registered", "IVLP_VL_Adapter_Prompt" in names)
chk("compute_frozen_class_subspace NOT registered",
    "compute_frozen_class_subspace" not in names)
obj = TRAINER_REGISTRY.get("IVLP_VL_Adapter_Prompt")
chk("registry resolves to a class, not a function", isinstance(obj, type))
chk("decorator sits directly on the class",
    any(b.__name__ == "TrainerDF" for b in obj.__mro__))
chk("helper still importable as a plain function",
    callable(T.compute_frozen_class_subspace)
    and not isinstance(T.compute_frozen_class_subspace, type))

print("\n" + ("TIER1 CHECK FAILED: " + ", ".join(FAIL) if FAIL else "ALL TIER 1 CHECKS PASSED"))
