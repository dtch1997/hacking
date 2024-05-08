#%%
import transformer_lens as tl

# %%
# MAIN
model = tl.HookedTransformer.from_pretrained('othello-gpt')
print(model)

# %% 
import datasets 
dataset = datasets.load_dataset("taufeeque/othellogpt")

# %% W
# print(dataset['train'][0])
print(model.embed.W_E.shape)

# %%
import torch
# from transformers import AutoTokenizer
model = model.to("cuda")
# HACK: we use the gpt-2 tokenizer
# And we convert tokens to token strs
# These will decode back to the desired tokens in the model
# tokenizer = AutoTokenizer.from_pretrained("gpt2")
# model.set_tokenizer(tokenizer)
# tokens = [20, 21, 34, 19]
# token_str = model.to_string(tokens)

# for i in range(5):
#     print(
#         model.generate(
#             token_str,
#             stop_at_eos=False,  # avoids a bug on MPS
#             temperature=1,
#             verbose=False,
#             max_new_tokens=50,
#         )
#     )

tokens = torch.LongTensor([[20, 21, 34, 19]]).to("cuda")
logits = model(tokens)
print(logits.shape)

# %%
import torch
import os
from sae_lens.training.config import LanguageModelSAERunnerConfig
from sae_lens.training.lm_runner import language_model_sae_runner

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Using device:", device)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


model_name = "othello-gpt"
layer: int = 6
hook_point = f"blocks.{layer}.hook_resid_post"
dataset_path = "taufeeque/othellogpt"
cfg = LanguageModelSAERunnerConfig(
    # Data Generating Function (Model + Training Distibuion)
    model_name=model_name,  # our model (more options here: https://neelnanda-io.github.io/TransformerLens/generated/model_properties_table.html)
    hook_point=hook_point,  # A valid hook point (see more details here: https://neelnanda-io.github.io/TransformerLens/generated/demos/Main_Demo.html#Hook-Points)
    hook_point_layer=layer,  # Only one layer in the model.
    d_in=512,  # the width of the mlp output.
    dataset_path=dataset_path,
    is_dataset_tokenized=True,
    # SAE Parameters
    mse_loss_normalization=None,  # We won't normalize the mse loss,
    expansion_factor=16,  # the width of the SAE. Larger will result in better stats but slower training.
    b_dec_init_method="geometric_median",  # The geometric median can be used to initialize the decoder weights.
    # Training Parameters
    lr=0.0008,  # lower the better, we'll go fairly high to speed up the tutorial.
    lr_scheduler_name="constant",  # constant learning rate with warmup. Could be better schedules out there.
    lr_warm_up_steps=10000,  # this can help avoid too many dead features initially.
    l1_coefficient=0.001,  # will control how sparse the feature activations are
    lp_norm=1.0,  # the L1 penalty (and not a Lp for p < 1)
    train_batch_size=4096,
    context_size=512,  # will control the lenght of the prompts we feed to the model. Larger is better but slower.
    # Activation Store Parameters
    n_batches_in_buffer=64,  # controls how many activations we store / shuffle.
    training_tokens=10_000,
    store_batch_size=16,
    # Resampling protocol
    use_ghost_grads=False,
    feature_sampling_window=1000,  # this controls our reporting of feature sparsity stats
    dead_feature_window=1000,  # would effect resampling or ghost grads if we were using it.
    dead_feature_threshold=1e-4,  # would effect resampling or ghost grads if we were using it.
    # WANDB
    log_to_wandb=False,  # always use wandb unless you are just testing code.
    wandb_project="sae_lens_tutorial",
    wandb_log_frequency=10,
    # Misc
    device=device,
    seed=42,
    n_checkpoints=0,
    checkpoint_path="checkpoints",
    dtype=torch.float32,
)

# look at the next cell to see some instruction for what to do while this is running.
sparse_autoencoder_dictionary = language_model_sae_runner(cfg)
# %%
