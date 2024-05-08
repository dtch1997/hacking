# flake8: noqa
# %%
import torch
import random
import collections
import transformer_lens as tl
import sae_lens as sl
from torch.optim import Adam
from jaxtyping import Float, Int

from torch.utils.data import DataLoader, TensorDataset

state2idx = {
    'S0': 0,
    'S1': 1,
    'SR': 2,
}

idx2state = {v: k for k, v in state2idx.items()}

def generate_z1r_sequence(length):
    """Generate a binary sequence of given length according to the Z1R process."""
    sequence = []
    sequence_state_idx = []
    state = random.choice(['S0', 'S1', 'SR'])

    for _ in range(length):
        if state == 'S0':
            next_state = 'S1'
            emit = '0'
        elif state == 'S1':
            next_state = 'SR'
            emit = '1'
        else:  # state == 'SR'
            next_state = 'S0'
            emit = str(random.randint(0, 1))
        
        sequence.append(emit)
        sequence_state_idx.append(state2idx[state])
        state = next_state
    
    return ''.join(sequence), sequence_state_idx

def get_msp_state(sequence):
    """Determine the Mixed-State Presentation (MSP) state for a given binary sequence."""
    if len(sequence) == 0:
        return 'n0'
    elif len(sequence) == 1:
        if sequence == '0':
            return 'n10'
        else:  # sequence == '1'
            return 'n11'
    else:  # len(sequence) >= 2
        if sequence[-2:] == '00':
            return 'S0'
        elif sequence[-2:] == '01':
            return 'S1'
        elif sequence[-2:] == '10':
            return 'n101'
        elif sequence[-2:] == '11':
            return 'S0'
        else:
            raise ValueError(f"Invalid sequence: {sequence}")

def get_dataset(
    n_samples: int = 10_000,
    seq_len: int = 10,
    print_debug_info: bool = False,
):
    data = []
    statess = []
    for _ in range(n_samples):
        seq, states = generate_z1r_sequence(seq_len)
        data.append(seq)
        statess.append(states)

    # Sanity check the states
    if print_debug_info:
        # Initial state distribution
        init_state_counts = collections.defaultdict(int)
        for states in statess:
            init_state_counts[states[0]] += 1

        print("Initial state distribution:")
        for k, v in init_state_counts.items():
            print(f"{k}: {v}")
        
        # Calculate statistics of transitions
        counts = collections.defaultdict(int)
        for states in statess:
            for i in range(len(states) - 1):
                counts[(states[i], states[i + 1])] += 1
        
        print("Transition counts:")
        for k, v in counts.items():
            print(f"{k}: {v}")

    data = [[int(c) for c in seq] for seq in data]
    data = torch.Tensor(data).to(torch.int64)
    dataset = TensorDataset(data)
    return dataset

def init_model(
    n_layers: int = 2,       
    seed: int = 0,
    d_model: int = 512,
    n_heads: int = 8,
):
    model_config=tl.HookedTransformerConfig(
        n_layers=n_layers,
        d_model=d_model,
        d_head= d_model // n_heads,
        n_heads=n_heads,
        d_mlp=256,
        d_vocab=2,
        n_ctx=10,
        normalization_type="LN",
        act_fn="relu",
        # attn_only=True,
        init_weights=True,
        device='cuda',
        # positional_embedding_type='rotary',
        seed=seed,
    )
    model = tl.HookedTransformer(model_config)
    return model

def loss_fn(
    logits: Float[torch.Tensor, "n_batch n_seq n_dim"], 
    tokens,
    loss_per_token: bool = False,
):
    """
    Calculate the negative log-likelihood loss for given logits and tokens.
    """
    logits = logits[:, :-1, :]
    tokens = tokens[:, 1:].long()
    log_probs = logits.log_softmax(dim=-1)
    correct_log_probs = log_probs.gather(dim=-1, index=tokens.unsqueeze(-1)).squeeze(-1)
    if loss_per_token:
        return -correct_log_probs
    else:
        return -correct_log_probs.mean()

def train_model(
    model: torch.nn.Module,
    train_data_loader: DataLoader,
    n_epochs: int = 100,
    learning_rate: float = 1e-3,
) -> list[float]:
    optim = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    model.train()

    losses = []
    for epoch in range(n_epochs):
        for batch in train_data_loader:
            x, = batch
            x = x.to("cuda")
            logits = model(x)
            # NOTE: index shift handled inside loss fn
            loss = loss_fn(logits, x)
            
            # Optimize
            model.zero_grad()
            loss.backward()
            optim.step()
            losses.append(loss.item())

        print(f"Epoch: {epoch} | loss: {loss.item():.3f}")
    return losses

# %%
dataset = get_dataset(n_samples = 100, seq_len = 10, print_debug_info=True)

# %%
results = {}

# %%
# Train a model

train_dataset = get_dataset(n_samples=10_000, seq_len=10)
train_data_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
model = init_model(d_model = 128)
model.to("cuda")
hist = train_model(
    model=model,
    train_data_loader=train_data_loader,
    n_epochs=10,
    learning_rate=1e-4,
)
results['train_model'] = hist


# %%

# Plot training results
import seaborn as sns 
import matplotlib.pyplot as plt
sns.set_theme()
hist = results['train_model']
sns.lineplot(x=range(len(hist)), y=hist)
# %%

# Test the model and plot per-token next-token loss
# We expect to see this decrease over token position
test_dataset = get_dataset(n_samples=1000, seq_len=10)
test_data_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

losses = []
for batch in test_data_loader:
    x, = batch
    x = x.to("cuda")
    logits = model(x)
    loss = loss_fn(logits, x, loss_per_token=True)
    loss = loss.mean(dim=0, keepdim=True)
    losses.append(loss)

losses = torch.cat(losses, dim=0)
mean_loss = losses.mean(dim=0).detach().cpu().numpy()
sns.lineplot(x=range(len(mean_loss)), y=mean_loss)
# %%
