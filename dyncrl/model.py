"""Stage 1: a temporal VAE with a regime-conditioned transition prior.

The model learns latents z_t from windows of observations x_{t-L..t}. Every
window carries a discrete regime label that selects the transition mechanism.
Both readouts in this repository (tracking.py and interpret.py) start from a
trained instance of this model and keep its encoder frozen.
"""

import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def mlp(sizes, slope=0.2):
    """Fully connected network with LeakyReLU between layers."""
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.LeakyReLU(slope))
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    """Frame-wise Gaussian encoder q(z_t | x_t)."""

    def __init__(self, x_dim, z_dim, hidden=128):
        super().__init__()
        self.net = mlp([x_dim, hidden, hidden, hidden, hidden, 2 * z_dim])

    def forward(self, x):
        mu, logvar = self.net(x).chunk(2, dim=-1)
        return mu, logvar


class DenseDecoder(nn.Module):
    """Unconstrained decoder used in Stage 1."""

    def __init__(self, z_dim, x_dim, hidden=128):
        super().__init__()
        self.net = mlp([z_dim, hidden, hidden, x_dim])

    def forward(self, z):
        return self.net(z)


class ParallelMLP(nn.Module):
    """n independent MLPs with scalar output, evaluated in one batched call."""

    def __init__(self, n, in_dim, hidden, layers=3, slope=0.2):
        super().__init__()
        dims = [in_dim] + [hidden] * layers + [1]
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for fan_in, fan_out in zip(dims[:-1], dims[1:]):
            bound = 1.0 / math.sqrt(fan_in)
            self.weights.append(nn.Parameter(torch.empty(n, fan_out, fan_in).uniform_(-bound, bound)))
            self.biases.append(nn.Parameter(torch.zeros(n, fan_out)))
        self.act = nn.LeakyReLU(slope)

    def forward(self, h):
        # h: (batch, n, in_dim) -> (batch, n)
        last = len(self.weights) - 1
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = torch.einsum("bni,noi->bno", h, w) + b
            if i < last:
                h = self.act(h)
        return h.squeeze(-1)


class TransitionPrior(nn.Module):
    """Regime-conditioned transition prior p(z_t | z_{t-L..t-1}, regime).

    Each latent i has its own network r_i = g_i(regime, past, z_{t,i}) that maps
    the current value to a noise residual. With standard normal noise the
    transition density is N(r; 0, I) * prod_i |d r_i / d z_{t,i}|.
    """

    def __init__(self, z_dim, lag, n_regimes, embed_dim=8, hidden=128, layers=3):
        super().__init__()
        self.z_dim, self.lag = z_dim, lag
        self.embed = nn.Embedding(n_regimes, embed_dim)
        self.context = mlp([embed_dim, hidden, hidden, hidden])
        self.flows = ParallelMLP(z_dim, hidden + lag * z_dim + 1, hidden, layers)

    def forward(self, z, regime):
        """z: (batch, T, n), regime: (batch,). Returns residuals (batch, T-L, n)
        and the summed log |det Jacobian| per sequence (batch,)."""
        batch, length, n = z.shape
        steps = length - self.lag
        windows = z.unfold(1, self.lag + 1, 1).permute(0, 1, 3, 2)  # (batch, steps, L+1, n)
        past = windows[:, :, :-1].reshape(batch * steps, self.lag * n)
        current = windows[:, :, -1].reshape(batch * steps, n)
        ctx = self.context(self.embed(regime))
        ctx = ctx.unsqueeze(1).expand(-1, steps, -1).reshape(batch * steps, -1)
        shared = torch.cat([ctx, past], dim=-1).unsqueeze(1).expand(-1, n, -1)

        with torch.enable_grad():
            if not current.requires_grad:
                current = current.detach().requires_grad_(True)
            residual = self.flows(torch.cat([shared, current.unsqueeze(-1)], dim=-1))
            # r_i depends on the current frame only through z_{t,i}, so the
            # gradient of sum_i r_i is the diagonal of the Jacobian.
            diag = torch.autograd.grad(residual.sum(), current, create_graph=self.training,
                                       retain_graph=True)[0]

        logdet = torch.log(diag.abs() + 1e-8).sum(-1).reshape(batch, steps).sum(1)
        return residual.reshape(batch, steps, n), logdet


class TemporalVAE(nn.Module):
    """Encoder, decoder and transition prior trained with a sequence ELBO."""

    def __init__(self, x_dim, z_dim, n_regimes, lag=2, hidden=128, embed_dim=8,
                 beta=2e-3, gamma=2e-2):
        super().__init__()
        self.x_dim, self.z_dim, self.lag = x_dim, z_dim, lag
        self.beta, self.gamma = beta, gamma
        self.encoder = Encoder(x_dim, z_dim, hidden)
        self.decoder = DenseDecoder(z_dim, x_dim, hidden)
        self.prior = TransitionPrior(z_dim, lag, n_regimes, embed_dim, hidden)

    def sample(self, x):
        mu, logvar = self.encoder(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return z, mu, logvar

    @torch.no_grad()
    def encode(self, x, batch_size=8192):
        """Posterior means for an array of any leading shape (..., x_dim)."""
        self.eval()
        device = next(self.parameters()).device
        x = torch.as_tensor(x, dtype=torch.float32)
        flat = x.reshape(-1, self.x_dim)
        out = [self.encoder(flat[i:i + batch_size].to(device))[0].cpu()
               for i in range(0, len(flat), batch_size)]
        return torch.cat(out).reshape(*x.shape[:-1], self.z_dim).numpy()

    def recon_loss(self, x, x_hat):
        """Squared error, with the frames after the first L averaged over time."""
        batch, length, _ = x.shape
        err = (x - x_hat) ** 2
        return (err[:, :self.lag].sum() + err[:, self.lag:].sum() / (length - self.lag)) / batch

    def elbo(self, x, regime):
        """x: (batch, T, x_dim), regime: (batch,). Returns the loss and its parts."""
        length = x.shape[1]
        z, mu, logvar = self.sample(x)
        recon = self.recon_loss(x, self.decoder(z))

        std = torch.exp(0.5 * logvar)
        log_q = torch.distributions.Normal(mu, std).log_prob(z)
        unit = torch.distributions.Normal(0.0, 1.0)

        # the first L frames have no history and use a standard normal prior
        kl_init = (log_q[:, :self.lag] - unit.log_prob(z[:, :self.lag])).sum((1, 2)).mean()

        # the remaining frames use the learned transition prior
        residual, logdet = self.prior(z, regime)
        log_p = unit.log_prob(residual).sum((1, 2)) + logdet
        kl_trans = ((log_q[:, self.lag:].sum((1, 2)) - log_p) / (length - self.lag)).mean()

        loss = recon + self.beta * kl_init + self.gamma * kl_trans
        return loss, {"recon": recon.item(), "kl_init": kl_init.item(), "kl_trans": kl_trans.item()}


def make_loader(x, regime, batch_size, shuffle=True):
    data = TensorDataset(torch.as_tensor(x, dtype=torch.float32),
                         torch.as_tensor(regime).reshape(-1).long())
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle)


def fit_stage1(model, x, regime, epochs=50, batch_size=256, lr=5e-4, device="cpu", log=print):
    """Train the temporal VAE on windows x (N, T, x_dim) with regime labels (N,)."""
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loader = make_loader(x, regime, batch_size)
    for epoch in range(epochs):
        model.train()
        total, parts, n = 0.0, {}, 0
        for xb, cb in loader:
            loss, info = model.elbo(xb.to(device), cb.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            for k, v in info.items():
                parts[k] = parts.get(k, 0.0) + v
            n += 1
        if log is not None:
            detail = ", ".join(f"{k} {v / n:.3f}" for k, v in parts.items())
            log(f"[stage 1] epoch {epoch + 1:3d}/{epochs}  loss {total / n:.3f}  ({detail})")
    return model
