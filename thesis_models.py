"""Shared model code for the thesis notebooks (AE and VAE for anomaly detection).

The AE and VAE are built from the same blocks so that the only difference between
them is the variational part: the Gaussian latent layer, the reparameterisation
trick and the KL term.
"""
import os, json, time, random
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mlp(sizes, dropout):
    """Dense layers with ReLU and dropout after every layer."""
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nn.ReLU(), nn.Dropout(dropout)]
    return nn.Sequential(*layers)


class AE(nn.Module):
    def __init__(self, n_in, hidden, latent, dropout=0.2):
        super().__init__()
        self.encoder = mlp([n_in, *hidden], dropout)
        self.to_latent = nn.Linear(hidden[-1], latent)
        self.decoder = mlp([latent, *hidden[::-1]], dropout)
        self.out = nn.Linear(hidden[0], n_in)

    def forward(self, x):
        z = self.to_latent(self.encoder(x))
        return self.out(self.decoder(z)), None, None

    def reconstruct(self, x):
        return self.forward(x)[0]


class VAE(nn.Module):
    def __init__(self, n_in, hidden, latent, dropout=0.2):
        super().__init__()
        self.encoder = mlp([n_in, *hidden], dropout)
        self.mu = nn.Linear(hidden[-1], latent)
        self.logvar = nn.Linear(hidden[-1], latent)
        self.decoder = mlp([latent, *hidden[::-1]], dropout)
        self.out = nn.Linear(hidden[0], n_in)
        # Start with a small sampling noise (log-variance -6, so a standard deviation of about 0.05).
        # With the default start (standard deviation 1) the noise drowns the signal, the decoder
        # learns to ignore z and the latent space collapses before it is ever used.
        nn.init.constant_(self.logvar.bias, -6.0)

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.mu(h), self.logvar(h).clamp(-10, 10)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)   # reparameterisation trick
        return self.out(self.decoder(z)), mu, logvar

    def reconstruct(self, x):
        """At scoring time the mean of q(z|x) is used, so the score is deterministic."""
        h = self.encoder(x)
        return self.out(self.decoder(self.mu(h)))


def loss_fn(x, x_hat, mu, logvar, beta):
    """Per-sample sum of squared errors (same for AE and VAE), plus beta * KL for the VAE."""
    rec = ((x_hat - x) ** 2).sum(dim=1).mean()
    if mu is None:
        return rec, rec, torch.zeros(())
    kl = (-0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(dim=1)).mean()   # closed-form KL
    return rec + beta * kl, rec, kl


@torch.no_grad()
def anomaly_score(model, X, batch=8192):
    """Mean squared reconstruction error per transaction (the anomaly score)."""
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).to(DEVICE)
        out.append(((model.reconstruct(xb) - xb) ** 2).mean(dim=1).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def active_units(model, X, n=20000, thresh=0.01):
    """VAE only: number of latent dimensions whose mean varies across inputs (posterior collapse check)."""
    model.eval()
    xb = torch.from_numpy(X[:n]).to(DEVICE)
    mu = model.mu(model.encoder(xb)).cpu().numpy()
    return int((mu.var(axis=0) > thresh).sum())


def train(kind, X_train, X_es, hidden, latent, seed, beta=0.01, lr=1e-3, batch=2048, max_epochs=150,
          es_patience=10, lr_patience=5, warmup=10, dropout=0.2, min_delta=1e-3, verbose=True):
    """beta is the KL weight of the VAE (beta-VAE). It is ignored for the AE."""
    beta_max = beta
    set_seed(seed)
    n_in = X_train.shape[1]
    model = (AE if kind == "AE" else VAE)(n_in, hidden, latent, dropout).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=lr_patience,
                                                       threshold=min_delta, threshold_mode="rel")
    Xt = torch.from_numpy(X_train).to(DEVICE)     # whole training set kept on the GPU when there is one
    Xe = torch.from_numpy(X_es).to(DEVICE)
    gen = torch.Generator().manual_seed(seed)

    hist = {"train_loss": [], "es_loss": [], "es_rec": [], "es_kl": [], "lr": [], "beta": []}
    best, best_state, wait, t0 = np.inf, None, 0, time.time()
    for ep in range(1, max_epochs + 1):
        # KL warm-up: the KL weight rises linearly from 0 to beta_max over the first epochs (VAE only)
        beta = beta_max * min(1.0, ep / warmup) if kind == "VAE" else 0.0
        model.train()
        perm = torch.randperm(len(Xt), generator=gen).to(DEVICE)
        tot, nb = 0.0, 0
        for i in range(0, len(Xt), batch):
            xb = Xt[perm[i:i + batch]]
            x_hat, mu, logvar = model(xb)
            loss, _, _ = loss_fn(xb, x_hat, mu, logvar, beta)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1

        # early-stopping set (normals only), always evaluated with the final KL weight
        model.eval()
        with torch.no_grad():
            r_sum, k_sum, m = 0.0, 0.0, 0
            for i in range(0, len(Xe), 8192):
                xb = Xe[i:i + 8192]
                x_hat, mu, logvar = model(xb) if kind == "VAE" else (model.reconstruct(xb), None, None)
                _, rec, kl = loss_fn(xb, x_hat, mu, logvar, beta_max)
                r_sum += rec.item() * len(xb); k_sum += kl.item() * len(xb); m += len(xb)
        es_rec, es_kl = r_sum / m, k_sum / m
        es_loss = es_rec + (beta_max * es_kl if kind == "VAE" else 0.0)
        sched.step(es_loss)

        for k, v in zip(hist, [tot / nb, es_loss, es_rec, es_kl, opt.param_groups[0]["lr"], beta]):
            hist[k].append(float(v))
        if verbose and (ep == 1 or ep % 5 == 0):
            print(f"  epoch {ep:3d} | train {tot/nb:.4f} | es {es_loss:.4f} (rec {es_rec:.4f}, kl {es_kl:.4f})"
                  f" | lr {opt.param_groups[0]['lr']:.1e} | {time.time()-t0:.0f}s")

        # an epoch counts as an improvement only if the loss drops by more than 0.1%
        improved = es_loss < best * (1 - min_delta)
        if es_loss < best:
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best = es_loss
        if improved:
            wait = 0
        elif kind == "AE" or ep > warmup:      # VAE: patience counts only after the warm-up
            wait += 1
            if wait >= es_patience:
                break

    model.load_state_dict(best_state)
    info = {"device": str(DEVICE), "epochs_run": ep, "best_epoch": int(np.argmin(hist["es_loss"]) + 1),
            "best_es_loss": float(best), "train_seconds": round(time.time() - t0, 1),
            "n_params": sum(p.numel() for p in model.parameters())}
    return model, hist, info


def run_experiment(ds, kind, data, hidden, latent, seed, out_root, tag="", **kw):
    """Train one configuration and save everything Notebook 3 needs. Skips runs already done."""
    beta = kw.get("beta", 0.01)
    name = f"{ds}_{kind}_z{latent}_s{seed}" if kind == "AE" else f"{ds}_{kind}_b{beta:g}_z{latent}_s{seed}"
    if tag:                                   # used by the sensitivity check, e.g. "_drop0"
        name += tag
    out = os.path.join(out_root, name)
    if os.path.exists(os.path.join(out, "meta.json")):
        with open(os.path.join(out, "meta.json")) as f:
            print(f"{name}: already done, skipping")
            return json.load(f)
    os.makedirs(out, exist_ok=True)
    print(f"{name}: training ...")
    model, hist, info = train(kind, data["X_train"], data["X_es"], hidden, latent, seed, **kw)

    s_es, s_val, s_test = (anomaly_score(model, data[k]) for k in ("X_es", "X_val", "X_test"))
    np.save(os.path.join(out, "scores_es.npy"), s_es.astype("float32"))
    np.save(os.path.join(out, "scores_val.npy"), s_val.astype("float32"))
    np.save(os.path.join(out, "scores_test.npy"), s_test.astype("float32"))
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, os.path.join(out, "model.pt"))
    with open(os.path.join(out, "history.json"), "w") as f:
        json.dump(hist, f)

    meta = {"dataset": ds, "model": kind, "latent": latent, "seed": seed, "hidden": list(hidden),
            "beta": beta if kind == "VAE" else None, "run": name, "tag": tag,
            "dropout": kw.get("dropout", 0.2), "lr": kw.get("lr", 1e-3), **info,
            # validation metrics, used only to choose the latent size (label-assisted, declared)
            "val_auc_pr": float(average_precision_score(data["y_val"], s_val)),
            "val_auc_roc": float(roc_auc_score(data["y_val"], s_val))}
    if kind == "VAE":
        meta["active_latent_units"] = active_units(model, data["X_es"])
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"{name}: done in {info['train_seconds']:.0f}s, {info['epochs_run']} epochs, "
          f"val AUC-PR {meta['val_auc_pr']:.4f}, val AUC-ROC {meta['val_auc_roc']:.4f}")
    return meta
