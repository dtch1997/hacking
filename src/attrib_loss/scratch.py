# flake8: noqa
# %%
import torch
import sys
sys.path.append("/workspace/smol-sae")
import torch.nn.functional as F
import transformer_lens as tl
import sae_lens as sl
from functools import partial
from torch import Tensor
from jaxtyping import Float
from smol_sae.base import Config
from smol_sae.utils import get_splits
from smol_sae.vanilla import VanillaSAE

from datasets import load_dataset 

# define loss function

# x [batch, d_model]
# grad_sae_acts [batch, ]
# assume gradients have a batch dim

def loss(
    x: Float[Tensor, "batch d_model"], 
    x_rec:  Float[Tensor, "batch d_model"], 
    sae_acts:  Float[Tensor, "batch d_sae"], 
    # backward hook
    grad_sae_acts:  Float[Tensor, "batch d_sae"],
    grad_x: Float[Tensor, "batch d_model"],
    lamda: float = 1.0,
    alpha: float = 1.0,
    beta: float = 1.0,
):
    # reconstruction term 
    l2_loss = (x-x_rec).square().sum()
    l1_loss = sae_acts.abs().sum()
    attr_loss = (sae_acts * grad_sae_acts).abs().sum()
    unexplained_loss = ((x-x_rec) * grad_x).abs().sum()
    
    return (
        l2_loss 
        + lamda * l1_loss 
        + alpha * attr_loss 
        + beta * unexplained_loss
    )
# %%
device = "cuda"
model = tl.HookedTransformer.from_pretrained("gelu-1l")
for hook_point in model.hook_points():
    print(hook_point.name)
# %%
train_dataset = load_dataset(
    "NeelNanda/c4-tokenized-2b", split="train", streaming=True
).with_format("torch")
train_batch = list(train_dataset.take(32))
print(train_batch)

# %%
config = Config(
    n_buffers=100, expansion=4, buffer_size=2**8, sparsities=(0.1, 1.0), device=device
)
sae = VanillaSAE(config, model)
print(sae.d_model)
print(model.cfg.d_model)

# %% 
print(sae.n_instances)
print(sae.W_dec.shape)
print(sae.W_enc.shape)
print(train_batch[0]["tokens"].shape)

# %%
import einops
import collections

tensors = collections.defaultdict(dict)

def fwd_patch_model_with_sae(
    act: Float[Tensor, "batch seq d_model"], 
    hook: tl.hook_points.HookPoint, 
    sae: VanillaSAE
):
    global tensors
    act_repeat: Float[Tensor, "batch seq inst d_model"] = einops.repeat(act, "batch seq d_model -> batch seq inst d_model", inst = sae.n_instances)
    sae_in = einops.rearrange(act_repeat, "batch seq inst d_model -> (batch seq) inst d_model")
    sae_hid: torch.Tensor = sae.encode(sae_in)[0]
    sae_out: torch.Tensor = sae.decode(sae_hid)
    sae_err =  sae_in - sae_out.detach()
    sae_rec: Float[Tensor, "batchseq inst d_model"] = sae_out + sae_err 
    act_rec = (sae_out + sae_err).view(*act_repeat.shape)
    sae_err.retain_grad()
    sae_hid.retain_grad()
    # mean along the instance dim
    tensors[hook.name]["x"] = act_repeat
    tensors[hook.name]["x_rec"] = act_rec
    tensors[hook.name]["sae_acts"] = sae_hid
    return act_rec.mean(dim=2)

def bwd_patch_model_gradient(grad_act, hook):
    global tensors 
    tensors[hook.name]["grad_x"] = grad_act.detach()
    sae_acts = tensors[hook.name]["sae_acts"]
    tensors[hook.name]["grad_sae_acts"] = sae_acts.grad
    # return (grad_act,)

model.reset_hooks()
model.set_use_hook_mlp_in(True)
with model.hooks(
    fwd_hooks=[("blocks.0.hook_mlp_in", partial(fwd_patch_model_with_sae, sae=sae))],
    bwd_hooks=[("blocks.0.hook_mlp_in", bwd_patch_model_gradient)],
):
    logits = model(train_batch[0]["tokens"].to(device))
    print(logits.shape)
    logits[0,0,0].backward()
    # sae(train_batch[0]["input_ids"])
# %%
for k, v in tensors['blocks.0.hook_mlp_in'].items():
    print(k, v.norm())

# %%
# Next steps: 
