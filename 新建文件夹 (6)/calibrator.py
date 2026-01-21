import argparse, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional deps for IRM
try:
    from sklearn.isotonic import IsotonicRegression
    _HAS_SK_ISO = True
except Exception:
    _HAS_SK_ISO = False

# Optional GNN ops for CaGCN/GATS
try:
    from torch_geometric.nn import GCNConv, GATConv
    _HAS_TG = True
except Exception:
    _HAS_TG = False

# -----------------------------
# Helpers
# -----------------------------
def _to_device(x, device):
    if isinstance(x, torch.Tensor): return x.to(device)
    return torch.tensor(x, device=device)

def _load_tensor(path, device="cpu"):
    if path is None: return None
    ext = os.path.splitext(path)[1].lower()
    if ext in [".pt", ".pth"]:
        t = torch.load(path, map_location=device)
        return t if isinstance(t, torch.Tensor) else torch.tensor(t, device=device)
    elif ext == ".npy":
        return torch.tensor(np.load(path), device=device)
    else:
        raise ValueError(f"Unsupported file: {path}")

def _as_index(mask_like, N=None, device="cpu"):
    """Accept bool mask or index tensor or python list → return 1D long index tensor."""
    if mask_like is None: return None
    if isinstance(mask_like, torch.Tensor):
        t = mask_like
    else:
        t = torch.tensor(mask_like)
    t = t.to(device)
    if t.dtype == torch.bool:
        return torch.nonzero(t, as_tuple=False).view(-1).long()
    return t.view(-1).long()

# -----------------------------
# Base API
# -----------------------------
class BaseCalibrator:
    name = "none"
    def __init__(self): self.ctx = {}
    def set_context(self, **ctx): self.ctx.update(ctx); return self
    def fit(self, logits_val: torch.Tensor, labels_val: torch.Tensor): return self
    def logits(self, logits_sub: torch.Tensor, mask=None) -> torch.Tensor: return logits_sub
    def probs(self, logits_sub: torch.Tensor, mask=None) -> torch.Tensor:
        return torch.softmax(self.logits(logits_sub, mask=mask), dim=1)
    def summary(self): return {}

# -----------------------------
# Classic post-hoc methods
# -----------------------------
class TemperatureScaling(BaseCalibrator):
    name = "ts"
    def __init__(self, T_init=1.0, lr=0.1, max_iter=200, patience=5):
        super().__init__()
        # 这里确保 T 是叶张量
        self.T = torch.tensor([T_init], requires_grad=True)

        self.lr, self.max_iter, self.patience = lr, max_iter, patience

    def fit(self, logits_val, labels_val):
        device = logits_val.device
        # 确保 T 是叶张量并将它放到正确的设备
        self.T = self.T.to(device).detach().requires_grad_()

        ce = nn.CrossEntropyLoss()
        opt = torch.optim.LBFGS([self.T], lr=self.lr, max_iter=self.max_iter, line_search_fn="strong_wolfe")
        best, best_T, noimp = float("inf"), self.T.detach().clone(), 0

        def closure():
            nonlocal best, best_T, noimp
            opt.zero_grad()
            loss = ce(logits_val / self.T.clamp(1e-3, 1e3), labels_val)
            loss.backward()
            if loss.item() < best - 1e-6:
                best, best_T, noimp = loss.item(), self.T.detach().clone(), 0
            else:
                noimp += 1
            return loss

        # Train T
        for _ in range(3):
            opt.step(closure)
            if noimp >= self.patience: break
        self.T = best_T.clamp(1e-2, 100)
        return self

    def logits(self, logits_sub, mask=None):
        return logits_sub / self.T

    def summary(self):
        return {"T": float(self.T.item())}


class ETS(BaseCalibrator):
    name = "ets"
    
    def __init__(self):
        super().__init__()
        self.ts = TemperatureScaling()
        self.alpha = nn.Parameter(torch.zeros(3))

    def fit(self, logits_val, labels_val):
        device = logits_val.device
        
        # 确保 `self.alpha` 在训练时保持可优化
        self.alpha = nn.Parameter(self.alpha.detach().to(device))
        
        # 确保 `TemperatureScaling` 中的 `T` 是叶张量
        self.ts.T = self.ts.T.detach().requires_grad_()
        
        self.ts.fit(logits_val, labels_val)

        ce = nn.CrossEntropyLoss()
        opt = torch.optim.LBFGS([self.alpha], lr=0.1, max_iter=200, line_search_fn="strong_wolfe")

        K = logits_val.size(1)
        p_raw = torch.softmax(logits_val.detach(), dim=1)
        p_ts  = torch.softmax(self.ts.logits(logits_val.detach()), dim=1)
        uni = torch.full_like(p_raw, 1.0 / K)

        def closure():
            opt.zero_grad()
            w = torch.softmax(self.alpha, dim=0)
            p = w[0] * p_raw + w[1] * p_ts + w[2] * uni
            loss = ce(torch.log(p.clamp_min(1e-12)), labels_val)
            loss.backward()
            return loss

        opt.step(closure)
        return self

    def probs(self, logits_sub, mask=None):
        K = logits_sub.size(1)
        p_raw = torch.softmax(logits_sub, dim=1)
        p_ts  = torch.softmax(self.ts.logits(logits_sub), dim=1)
        uni = torch.full_like(p_raw, 1.0 / K)
        w = torch.softmax(self.alpha, dim=0)
        return (w[0] * p_raw + w[1] * p_ts + w[2] * uni).clamp_min(1e-12)

    def summary(self):
        w = torch.softmax(self.alpha, dim=0).detach().cpu().numpy().tolist()
        return {
            "w_raw": float(w[0]),
            "w_ts": float(w[1]),
            "w_uniform": float(w[2]),
            "T_ts": float(self.ts.T.item())
        }


# -----------------------------
# Additional post-hoc methods
# -----------------------------
class VectorScaling(BaseCalibrator):
    name = "vs"
    def __init__(self): super().__init__(); self.a=None; self.b=None
    def fit(self, logits_val, labels_val):
        device = logits_val.device; K = logits_val.size(1)
        self.a = nn.Parameter(torch.ones(K, device=device))
        self.b = nn.Parameter(torch.zeros(K, device=device))
        ce = nn.CrossEntropyLoss()
        opt = torch.optim.LBFGS([self.a, self.b], lr=0.1, max_iter=300, line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad()
            z = logits_val * self.a.unsqueeze(0) + self.b.unsqueeze(0)
            loss = ce(z, labels_val); loss.backward(); return loss
        opt.step(closure); return self
    def logits(self, logits_sub, mask=None):
        return logits_sub * self.a.unsqueeze(0) + self.b.unsqueeze(0)
    def summary(self): return {"a_mean": float(self.a.mean().item()), "b_mean": float(self.b.mean().item())}

class DirichletCalibrator(BaseCalibrator):
    name = "dirichlet"
    def __init__(self, **kwargs):  # 接受并忽略多余参数
        super().__init__()
        self.W=None
    def fit(self, logits_val, labels_val):
        device = logits_val.device
        p = torch.softmax(logits_val.detach(), dim=1); N,K = p.size()
        self.W = nn.Parameter(torch.zeros(K,3, device=device))
        ce = nn.CrossEntropyLoss()
        opt = torch.optim.LBFGS([self.W], lr=0.1, max_iter=200, line_search_fn="strong_wolfe")
        eps = 1e-12
        def closure():
            opt.zero_grad()
            logp = torch.log(p.clamp_min(eps))
            log1mp = torch.log((1-p).clamp_min(eps))
            ones = torch.ones_like(p)
            feats = torch.stack([logp, log1mp, ones], dim=-1)  # (N,K,3)
            z = (feats * self.W.unsqueeze(0)).sum(dim=-1)
            loss = ce(z, labels_val); loss.backward(); return loss
        opt.step(closure); return self
    def logits(self, logits_sub, mask=None):
        p = torch.softmax(logits_sub, dim=1).clamp_min(1e-12)
        logp = torch.log(p); log1mp = torch.log((1-p).clamp_min(1e-12))
        ones = torch.ones_like(p)
        feats = torch.stack([logp, log1mp, ones], dim=-1)
        return (feats * self.W.unsqueeze(0)).sum(dim=-1)

class IRMCalibrator(BaseCalibrator):
    name = "irm"
    def __init__(self): super().__init__(); self.models=None
    def fit(self, logits_val, labels_val):
        if not _HAS_SK_ISO:
            raise RuntimeError("IRM requires scikit-learn: from sklearn.isotonic import IsotonicRegression")
        p = torch.softmax(logits_val.detach(), dim=1).cpu().numpy()
        y = labels_val.detach().cpu().numpy(); K = p.shape[1]
        self.models = []
        for k in range(K):
            m = IsotonicRegression(y_min=1e-6, y_max=1-1e-6, out_of_bounds='clip')
            m.fit(p[:,k], (y==k).astype(np.float32))
            self.models.append(m)
        return self
    def probs(self, logits_sub, mask=None):
        p = torch.softmax(logits_sub, dim=1).detach().cpu().numpy()
        K = p.shape[1]; cal = np.zeros_like(p)
        for k in range(K):
            cal[:,k] = self.models[k].predict(p[:,k])
        s = cal.sum(axis=1, keepdims=True); s[s==0]=1.0
        cal = cal / s
        return torch.tensor(cal, device=logits_sub.device, dtype=logits_sub.dtype).clamp_min(1e-12)

# -----------------------------
# Graph-aware: CaGCN / GATS
# -----------------------------
class _TempGCN(nn.Module):
    def __init__(self, in_dim, hidden=32):
        super().__init__()
        self.g1 = GCNConv(in_dim, hidden)
        self.g2 = GCNConv(hidden, hidden)
        self.out = nn.Linear(hidden, 1)
    def forward(self, x, edge_index):
        h = F.relu(self.g1(x, edge_index))
        h = F.relu(self.g2(h, edge_index))
        t = self.out(h).squeeze(-1)
        return F.softplus(t) + 1e-3

class _TempGAT(nn.Module):
    def __init__(self, in_dim, hidden=32, heads=4):
        super().__init__()
        self.g1 = GATConv(in_dim, hidden//heads, heads=heads, concat=True)
        self.g2 = GATConv(hidden, hidden//heads, heads=heads, concat=True)
        self.out = nn.Linear(hidden, 1)
    def forward(self, x, edge_index):
        h = F.relu(self.g1(x, edge_index))
        h = F.relu(self.g2(h, edge_index))
        t = self.out(h).squeeze(-1)
        return F.softplus(t) + 1e-3

class _TempGraphBase(BaseCalibrator):
    def __init__(self, hidden=32, lr=1e-2, wd=0.0, epochs=200, patience=30, use_feats=True):
        super().__init__()
        self.hidden, self.lr, self.wd = hidden, lr, wd
        self.epochs, self.patience, self.use_feats = epochs, patience, use_feats
        self.net = None
    def _build_input(self, x, logits_all):
        feats = [logits_all.detach()]
        if self.use_feats and x is not None: feats = [x, logits_all.detach()]
        return torch.cat(feats, dim=1)
    def _make(self, in_dim): raise NotImplementedError
    def fit(self, logits_val, labels_val):
        if not _HAS_TG:
            raise RuntimeError("CaGCN/GATS require torch_geometric installed.")
        x = self.ctx.get("x", None)
        edge_index = self.ctx["edge_index"]
        logits_all = self.ctx["logits_all"]
        val_idx = _as_index(self.ctx["valid_idx"], device=edge_index.device)
        inp = self._build_input(x, logits_all)
        in_dim = inp.size(1)
        self.net = self._make(in_dim).to(edge_index.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        ce = nn.CrossEntropyLoss()
        best, best_state, noimp = float("inf"), None, 0
        for ep in range(1, self.epochs+1):
            self.net.train(); opt.zero_grad()
            T_all = self.net(inp, edge_index)  # (N,)
            z_val = logits_all[val_idx] / T_all[val_idx].unsqueeze(-1).clamp(1e-3, 1e3)
            loss = ce(z_val, labels_val)
            loss.backward(); opt.step()
            if loss.item() < best - 1e-5: best, best_state, noimp = loss.item(), self.net.state_dict(), 0
            else: noimp += 1
            if noimp >= self.patience: break
        if best_state is not None: self.net.load_state_dict(best_state)
        return self
    def probs(self, logits_sub, mask=None):
        edge_index = self.ctx["edge_index"]
        logits_all = self.ctx["logits_all"]
        x = self.ctx.get("x", None)
        inp = self._build_input(x, logits_all)
        with torch.no_grad():
            T_all = self.net(inp, edge_index).clamp(1e-3, 1e3)
        idx = _as_index(mask, device=logits_all.device)
        if idx is None:
            T = T_all[:logits_sub.size(0)].unsqueeze(-1)
        else:
            T = T_all[idx].unsqueeze(-1)
        return torch.softmax(logits_sub / T, dim=1)

class CaGCNCalibrator(_TempGraphBase):
    name = "cagcn"
    def _make(self, in_dim): return _TempGCN(in_dim, hidden=self.hidden)

class GATSCalibrator(_TempGraphBase):
    name = "gats"
    def _make(self, in_dim): return _TempGAT(in_dim, hidden=self.hidden)

# -----------------------------
# Order-Invariant / Spline
# -----------------------------
class OrderInvariantCalibrator(BaseCalibrator):
    name = "orderinvariant"
    def __init__(self, hidden=32, lr=1e-2, epochs=200, patience=30):
        super().__init__()
        self.hidden, self.lr, self.epochs, self.patience = hidden, lr, epochs, patience
        self.mlp = None
    def _stats(self, logits):
        p = torch.softmax(logits.detach(), dim=1)
        max_logit, _ = logits.max(dim=1)
        top2, _ = torch.topk(logits, k=2, dim=1)
        margin = (top2[:,0] - top2[:,1]).detach()
        entropy = (-p * p.clamp_min(1e-12).log()).sum(dim=1)
        return torch.stack([max_logit.detach(), margin, entropy], dim=1)
    def fit(self, logits_val, labels_val):
        device = logits_val.device
        feats = self._stats(logits_val).to(device)
        self.mlp = nn.Sequential(
            nn.Linear(feats.size(1), self.hidden), nn.ReLU(),
            nn.Linear(self.hidden, 1)
        ).to(device)
        opt = torch.optim.Adam(self.mlp.parameters(), lr=self.lr)
        ce = nn.CrossEntropyLoss()
        best, best_state, noimp = float("inf"), None, 0
        for ep in range(1, self.epochs+1):
            self.mlp.train(); opt.zero_grad()
            T = F.softplus(self.mlp(feats)).squeeze(-1) + 1e-3
            z = logits_val / T.unsqueeze(-1).clamp(1e-3, 1e3)
            loss = ce(z, labels_val); loss.backward(); opt.step()
            if loss.item() < best - 1e-5: best, best_state, noimp = loss.item(), self.mlp.state_dict(), 0
            else: noimp += 1
            if noimp >= self.patience: break
        if best_state is not None: self.mlp.load_state_dict(best_state)
        return self
    def probs(self, logits_sub, mask=None):
        T = F.softplus(self.mlp(self._stats(logits_sub))).squeeze(-1) + 1e-3
        return torch.softmax(logits_sub / T.unsqueeze(-1), dim=1)

class SplineCalibrator(BaseCalibrator):
    name = "spline"
    def __init__(self, num_knots=10, lr=1e-2, epochs=200, patience=30):
        super().__init__()
        self.num_knots, self.lr, self.epochs, self.patience = num_knots, lr, epochs, patience
        self.delta = None
    def _init_params(self, device): self.delta = nn.Parameter(torch.zeros(self.num_knots+1, device=device))
    def _xgrid(self, device): return torch.linspace(0., 1., self.num_knots+2, device=device)
    def _ygrid(self):
        inc = F.softplus(self.delta) + 1e-6
        y = torch.cumsum(inc, dim=0)
        y = torch.cat([torch.zeros(1, device=y.device), y], dim=0)
        return y / y[-1].clamp_min(1e-6)
    def _g(self, p, xg, yg):
        idx = torch.bucketize(p.clamp(0,1), xg) - 1
        idx = idx.clamp(min=0, max=xg.numel()-2)
        x0 = xg[idx]; x1 = xg[idx+1]; y0 = yg[idx]; y1 = yg[idx+1]
        t = (p - x0) / (x1 - x0 + 1e-12)
        return y0 + t * (y1 - y0)
    def fit(self, logits_val, labels_val):
        device = logits_val.device; self._init_params(device)
        opt = torch.optim.Adam([self.delta], lr=self.lr)
        best, best_state, noimp = float("inf"), None, 0
        xg = self._xgrid(device)
        ce = nn.CrossEntropyLoss()
        for ep in range(1, self.epochs+1):
            yg = self._ygrid()
            p = torch.softmax(logits_val.detach(), dim=1).clamp(1e-12, 1-1e-12)
            p_cal = self._g(p, xg, yg)
            p_cal = (p_cal / p_cal.sum(dim=1, keepdim=True).clamp_min(1e-12)).clamp_min(1e-12)
            loss = ce(torch.log(p_cal), labels_val)
            opt.zero_grad(); loss.backward(); opt.step()
            if loss.item() < best - 1e-6: best, best_state, noimp = loss.item(), self.delta.detach().clone(), 0
            else: noimp += 1
            if noimp >= self.patience: break
        if best_state is not None: self.delta = nn.Parameter(best_state)
        return self
    def probs(self, logits_sub, mask=None):
        device = logits_sub.device
        xg = self._xgrid(device); yg = self._ygrid()
        p = torch.softmax(logits_sub, dim=1).clamp(1e-12, 1-1e-12)
        p_cal = self._g(p, xg, yg)
        p_cal = (p_cal / p_cal.sum(dim=1, keepdim=True).clamp_min(1e-12)).clamp_min(1e-12)
        return p_cal

# -----------------------------
# Factory & Convenience funcs
# -----------------------------
def make_calibrator(name: str, **kwargs) -> BaseCalibrator:
    """Create a calibrator; only pass supported kwargs to each method."""
    def _only(d, keys):
        return {k: d[k] for k in keys if k in d}

    n = (name or "none").lower()
    if n in ["none", "off"]:
        return BaseCalibrator()

    if n == "ts":
        return TemperatureScaling(**_only(kwargs, ["T_init", "lr", "max_iter", "patience"]))

    if n == "ets":
        return ETS()

    if n in ["vs", "vectorscaling"]:
        return VectorScaling()

    if n in ["dir", "dirichlet"]:
        return DirichletCalibrator()  # 忽略 CLI 超参

    if n in ["irm", "isotonic", "mcir"]:
        return IRMCalibrator()

    if n == "cagcn":
        return CaGCNCalibrator(**_only(kwargs, ["hidden", "lr", "wd", "epochs", "patience", "use_feats"]))

    if n == "gats":
        return GATSCalibrator(**_only(kwargs, ["hidden", "lr", "wd", "epochs", "patience", "use_feats"]))

    if n in ["orderinvariant", "oi"]:
        return OrderInvariantCalibrator(**_only(kwargs, ["hidden", "lr", "epochs", "patience"]))

    if n == "spline":
        return SplineCalibrator(**_only(kwargs, ["num_knots", "lr", "epochs", "patience"]))

    raise ValueError(f"Unknown posthoc method: {name}")

def fit_calibrator_from_val(cal: BaseCalibrator,
                            logits_all: torch.Tensor,
                            labels: torch.Tensor,
                            valid_idx,
                            context: dict = None) -> BaseCalibrator:
    """Fit on validation split. `context` can include x, edge_index, logits_all, valid_idx, etc."""
    if context: cal.set_context(**context)
    vidx = _as_index(valid_idx, device=logits_all.device)
    cal.fit(logits_all[vidx], labels[vidx])
    return cal

def apply_calibrator(cal: BaseCalibrator,
                     logits_all: torch.Tensor,
                     mask=None) -> torch.Tensor:
    """Return calibrated probabilities on `mask` (indices or bool-mask)."""
    idx = _as_index(mask, device=logits_all.device)
    logits_sub = logits_all if idx is None else logits_all[idx]
    # ensure no grad & detached
    with torch.no_grad():
        probs = cal.probs(logits_sub, mask=idx)
    return probs.detach()

# -----------------------------
# CLI
# -----------------------------
def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--method", type=str, default="ts",
                   choices=["none", "ts", "ets", "vs", "dirichlet", "irm", "cagcn", "gats", "orderinvariant", "spline"])
    p.add_argument("--logits_all", type=str, required=True, help="Tensor file (.pt/.pth/.npy) of shape [N,C]")
    p.add_argument("--labels", type=str, required=True, help="Tensor file (.pt/.pth/.npy) of shape [N]")
    p.add_argument("--valid_idx", type=str, required=True, help="Index/Bool mask for validation")
    p.add_argument("--mask", type=str, required=True, help="Index/Bool mask to output probs for")
    p.add_argument("--out", type=str, required=True, help="Where to save probs (pt or npy)")

    # optional graph context
    p.add_argument("--x", type=str, default=None, help="Node features tensor path [N,d]")
    p.add_argument("--edge_index", type=str, default=None, help="Edge index tensor path [2,E]")

    # optional hyperparams
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    logits_all = _load_tensor(args.logits_all, device)
    labels = _load_tensor(args.labels, device).long().view(-1)
    valid_idx = _load_tensor(args.valid_idx, device)
    mask = _load_tensor(args.mask, device)

    cal = make_calibrator(args.method, hidden=args.hidden, lr=args.lr, epochs=args.epochs, patience=args.patience)

    ctx = {"logits_all": logits_all}
    if args.x is not None: ctx["x"] = _load_tensor(args.x, device)
    if args.edge_index is not None: ctx["edge_index"] = _load_tensor(args.edge_index, device).long()
    if ctx: ctx["valid_idx"] = valid_idx

    fit_calibrator_from_val(cal, logits_all, labels, valid_idx, context=ctx)
    probs = apply_calibrator(cal, logits_all, mask=mask).detach().cpu()

    ext = os.path.splitext(args.out)[1].lower()
    if ext in [".pt", ".pth"]:
        torch.save(probs, args.out)
    elif ext == ".npy":
        np.save(args.out, probs.numpy())
    else:
        raise ValueError("Output must be .pt/.pth or .npy")
    print(f"[OK] Saved calibrated probabilities to {args.out}")

if __name__ == "__main__":
    _cli()




