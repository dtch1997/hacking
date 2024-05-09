# flake8: noqa
# %%
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
    sae, 
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

gradients = {}

def fwd_patch_model_with_sae(act, hook, sae):
    sae_out, hidden = sae(act)[:2]
    sae_err = act - sae_out.detach()
    return sae_out + sae_err 

def bwd_patch_model_gradient(grad_act, hook):
    global gradients 
    gradients[hook.name] = grad_act.detach()
    return grad_act

# %%

device = "cuda"
model = tl.HookedTransformer.from_pretrained("gelu-1l")
print(model.hooks)
config = Config(
    n_buffers=100, expansion=4, buffer_size=2**8, sparsities=(0.1, 1.0), device=device
)
sae = VanillaSAE(config, model)

train_dataset = load_dataset(
    "NeelNanda/c4-tokenized-2b", split="train", streaming=True
).with_format("torch")
train_batch = list(train_dataset.take(32))

model.run_with_hooks(
    train_batch,
    fwd_hooks=[("", partial(fwd_patch_model_with_sae, sae=sae))],
    bwd_hooks=[("", bwd_patch_model_gradient)],
)

# %%

