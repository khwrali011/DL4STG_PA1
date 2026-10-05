from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


HORIZON = 168
CONTEXT = 672
LABEL = 168


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(folder: Path):
    train = pd.read_csv(folder / "student_train.csv")
    test = pd.read_csv(folder / "student_test.csv")
    ext = pd.read_csv(folder / "optional_external_data.csv")
    assert list(train.columns) == ["time_idx", "value"]
    assert list(test.columns) == ["time_idx", "value"]
    assert np.array_equal(train.time_idx.to_numpy(), np.arange(1, 43657))
    assert np.array_equal(test.time_idx.to_numpy(), np.arange(43657, 43825))
    assert np.array_equal(ext.time_idx.to_numpy(), np.arange(1, 43825))
    assert train.value.notna().all() and test.value.isna().all()
    assert ext.notna().all().all()
    y = train.value.to_numpy(dtype=np.float32)
    x = ext.drop(columns="time_idx").to_numpy(dtype=np.float32)
    assert (y >= 0).all()
    return y, x, test


class Scaler:
    def __init__(self, y, ext, fit_end):
        
        t = np.log1p(y[:fit_end].astype(np.float64))
        self.y_mean, self.y_std = float(t.mean()), float(max(t.std(), 1e-5))
        e = ext.astype(np.float64).copy()
        e[:, 4:6] = np.log1p(np.maximum(e[:, 4:6], 0))
        self.e_mean = e[:fit_end].mean(axis=0)
        self.e_std = np.maximum(e[:fit_end].std(axis=0), 1e-5)
        
        self.e_mean[6:] = 0
        self.e_std[6:] = 1

    def transform(self, y, ext):
        yy = (np.log1p(y) - self.y_mean) / self.y_std
        ee = ext.astype(np.float64).copy()
        ee[:, 4:6] = np.log1p(np.maximum(ee[:, 4:6], 0))
        ee = (ee - self.e_mean) / self.e_std
        return yy.astype(np.float32), ee.astype(np.float32)

    def inverse(self, z):
        return np.maximum(np.expm1(np.clip(z * self.y_std + self.y_mean, -20, 20)), 0)


class Windows(Dataset):
    def __init__(self, y, ext, origins):
        self.y, self.ext = y, ext
        self.origins = np.asarray(origins, dtype=np.int64)

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, i):
        o = int(self.origins[i])  
        return (torch.from_numpy(self.y[o-CONTEXT:o]),
                torch.from_numpy(self.ext[o-CONTEXT:o+HORIZON]),
                torch.from_numpy(self.y[o:o+HORIZON]))


class Decomposition(nn.Module):
    def __init__(self, kernel=25):
        super().__init__()
        assert kernel % 2 == 1
        self.kernel = kernel

    def forward(self, x):
        z = x.transpose(1, 2)
        p = self.kernel // 2
        trend = F.avg_pool1d(F.pad(z, (p, p), mode="replicate"),
                             self.kernel, stride=1).transpose(1, 2)
        return x - trend, trend


class AutoCorrelation(nn.Module):
    """FFT series-wise lag scores and weighted circular time-delay aggregation."""
    def __init__(self, width, heads=4, top_k=5):
        super().__init__()
        assert width % heads == 0
        self.heads, self.dim, self.top_k = heads, width // heads, top_k
        self.q = nn.Linear(width, width)
        self.k = nn.Linear(width, width)
        self.v = nn.Linear(width, width)
        self.out = nn.Linear(width, width)

    def forward(self, query, key_value):
        b, t, width = query.shape
        
        
        if key_value.size(1) < t:
            key_value = F.pad(key_value.transpose(1, 2),
                              (0, t-key_value.size(1))).transpose(1, 2)
        else:
            
            
            key_value = key_value[:, -t:]
        q = self.q(query).view(b, t, self.heads, self.dim)
        k = self.k(key_value).view(b, t, self.heads, self.dim)
        v = self.v(key_value).view(b, t, self.heads, self.dim)
        
        scores = torch.fft.irfft(
            torch.fft.rfft(q.float(), dim=1) *
            torch.conj(torch.fft.rfft(k.float(), dim=1)), n=t, dim=1)
        scores = scores.mean(dim=(2, 3)) / (math.sqrt(self.dim) * t)
        weights, delays = scores.topk(min(self.top_k, t), dim=1)
        weights = weights.softmax(dim=1).to(v.dtype)
        positions = torch.arange(t, device=v.device)[None, None, :]
        positions = (positions - delays[:, :, None]) % t
        
        vv = v.permute(0, 2, 3, 1).reshape(b, width, t)
        picked = vv[:, None].expand(-1, delays.size(1), -1, -1).gather(
            3, positions[:, :, None, :].expand(-1, -1, width, -1))
        mixed = (picked * weights[:, :, None, None]).sum(1)
        return self.out(mixed.transpose(1, 2))


class EncoderLayer(nn.Module):
    def __init__(self, width, kernel, top_k):
        super().__init__()
        self.auto = AutoCorrelation(width, top_k=top_k)
        self.decomp = Decomposition(kernel)
        self.ff = nn.Sequential(nn.Linear(width, 2*width), nn.GELU(),
                                nn.Linear(2*width, width))
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)

    def forward(self, x):
        x, _ = self.decomp(self.norm1(x + self.auto(x, x)))
        x, _ = self.decomp(self.norm2(x + self.ff(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, width, kernel, top_k):
        super().__init__()
        self.self_auto = AutoCorrelation(width, top_k=top_k)
        self.cross_auto = AutoCorrelation(width, top_k=top_k)
        self.decomp = Decomposition(kernel)
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(3)])
        self.ff = nn.Sequential(nn.Linear(width, 2*width), nn.GELU(),
                                nn.Linear(2*width, width))
        self.trend_out = nn.Linear(width, 1)

    def forward(self, x, memory):
        x, t1 = self.decomp(self.norms[0](x + self.self_auto(x, x)))
        x, t2 = self.decomp(self.norms[1](x + self.cross_auto(x, memory)))
        x, t3 = self.decomp(self.norms[2](x + self.ff(x)))
        return x, self.trend_out(t1 + t2 + t3).squeeze(-1)


class SmallAutoformer(nn.Module):
    def __init__(self, n_external, width=32, kernel=25, top_k=5):
        super().__init__()
        self.decomp = Decomposition(kernel)
        self.enc_in = nn.Linear(1+n_external, width)
        self.dec_in = nn.Linear(1+n_external, width)
        self.encoder = nn.ModuleList([EncoderLayer(width, kernel, top_k)])
        self.decoder = nn.ModuleList([DecoderLayer(width, kernel, top_k)])
        self.season_out = nn.Linear(width, 1)

    def forward(self, history, external):
        assert history.shape[1] == CONTEXT and external.shape[1] == CONTEXT+HORIZON
        seasonal, trend = self.decomp(history.unsqueeze(-1))
        n = external.shape[-1]
        enc = self.enc_in(torch.cat([history.unsqueeze(-1), external[:, :CONTEXT]], -1))
        for layer in self.encoder:
            enc = layer(enc)
        initial_season = torch.cat([seasonal[:, -LABEL:],
                                    history.new_zeros((len(history), HORIZON, 1))], 1)
        future_trend = trend.mean(1, keepdim=True).expand(-1, HORIZON, -1)
        initial_trend = torch.cat([trend[:, -LABEL:], future_trend], 1).squeeze(-1)
        dec_external = external[:, CONTEXT-LABEL:]
        assert dec_external.shape[1] == LABEL+HORIZON
        dec = self.dec_in(torch.cat([initial_season, dec_external], -1))
        adjustment = initial_trend.new_zeros(initial_trend.shape)
        for layer in self.decoder:
            dec, add = layer(dec, enc)
            adjustment = adjustment + add
        return (initial_trend + adjustment + self.season_out(dec).squeeze(-1))[:, -HORIZON:]


def metrics(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    return {"mae": float(np.mean(np.abs(actual-predicted))),
            "rmse": float(np.sqrt(np.mean((actual-predicted)**2))),
            "smape": float(np.mean(200*np.abs(actual-predicted) /
                                    np.maximum(np.abs(actual)+np.abs(predicted), 1e-8)))}


@torch.no_grad()
def evaluate(model, loader, scaler, device):
    model.eval()
    actual, predicted = [], []
    for history, ext, future in loader:
        forecast = model(history.to(device), ext.to(device)).cpu().numpy()
        predicted.append(scaler.inverse(forecast))
        actual.append(scaler.inverse(future.numpy()))
    actual, predicted = np.concatenate(actual), np.concatenate(predicted)
    result = metrics(actual, predicted)
    result["block_rmse"] = np.sqrt(np.mean((actual-predicted)**2, axis=1)).tolist()
    return result


def train_run(y, ext, fit_end, val_origins, seed, use_external, epochs,
              batch_size, stride, device, width, kernel, top_k, patience,
              fixed_epochs=None, loss_name="log_mse"):
    seed_all(seed)
    scaler = Scaler(y, ext, fit_end)
    yn, en = scaler.transform(y, ext)
    if not use_external:
        en = en[:, :0].copy()
    origins = np.arange(CONTEXT, fit_end-HORIZON+1, stride)
    training = DataLoader(Windows(yn, en, origins), batch_size=batch_size,
                          shuffle=True, num_workers=0)
    validation = None
    if val_origins is not None:
        validation = DataLoader(Windows(yn, en, val_origins),
                                batch_size=batch_size, shuffle=False)
    model = SmallAutoformer(en.shape[1], width, kernel, top_k).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    best_state, best_score, best_epoch, stale = None, float("inf"), 0, 0
    history_log = []
    for epoch in range(1, (fixed_epochs or epochs)+1):
        model.train()
        losses = []
        for past, cov, future in training:
            past, cov, future = past.to(device), cov.to(device), future.to(device)
            optimizer.zero_grad(set_to_none=True)
            forecast = model(past, cov)
            loss_log = F.mse_loss(forecast, future)
            if loss_name == "log_mse":
                loss = loss_log
            else:
                
                
                
                raw_pred = torch.expm1(torch.clamp(
                    forecast*scaler.y_std+scaler.y_mean, min=-10, max=9))
                raw_true = torch.expm1(future*scaler.y_std+scaler.y_mean)
                raw_loss = F.mse_loss(raw_pred/100.0, raw_true/100.0)
                if loss_name == "hybrid_raw_log":
                    loss = 0.7*raw_loss + 0.3*loss_log
                elif loss_name == "raw_mse":
                    loss = raw_loss
                else:
                    raise ValueError(loss_name)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        record = {"epoch": epoch, "train_optimization_loss": float(np.mean(losses))}
        if validation is not None:
            record.update(evaluate(model, validation, scaler, device))
            if record["rmse"] < best_score:
                best_score, best_epoch, stale = record["rmse"], epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
        history_log.append(record)
        print(f"loss={loss_name} context={CONTEXT} external={use_external} "
              f"seed={seed} epoch={epoch} train={record['train_optimization_loss']:.4f} "
              f"val_rmse={record.get('rmse', float('nan')):.3f}", flush=True)
        if validation is not None and stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, scaler, history_log, (best_epoch or len(history_log))


def write_progress(path, signature, records):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"signature": signature, "records": records}, indent=2))
    temporary.replace(path)


def load_previous(path):
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            matches = [name for name in archive.namelist()
                       if name.endswith("/search_progress.json")
                       or name == "search_progress.json"]
            if len(matches) != 1:
                raise ValueError("Expected one search_progress.json in the prior ZIP")
            return json.loads(archive.read(matches[0]))
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=Path("Question 2 - Leaderboard/Data"))
    parser.add_argument("--out", type=Path, default=Path("question2_672_ablation_output"))
    parser.add_argument("--previous-search", type=Path,
                        default=Path("question2_search_output/search_progress.json"))
    parser.add_argument("--no-reuse", action="store_true",
                        help="Train both ablation arms instead of reusing compatible earlier runs")
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 29, 43])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--kernel", type=int, default=25)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    if len(set(args.seeds)) < 2:
        raise ValueError("The assignment requires multiple seeds for this ablation.")
    y, external, _ = load_data(args.data_dir)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    folds = [(27000, 33000), (33000, len(y))]
    signature = {"seeds": args.seeds, "epochs": args.epochs,
                 "patience": args.patience, "batch_size": args.batch_size,
                 "stride": args.stride, "width": args.width,
                 "kernel": args.kernel, "top_k": args.top_k,
                 "context": CONTEXT, "loss": "hybrid_raw_log",
                 "folds": [list(f) for f in folds]}
    args.out.mkdir(parents=True, exist_ok=True)
    progress_file = args.out / "ablation_progress.json"
    if progress_file.exists():
        progress = json.loads(progress_file.read_text())
        if progress["signature"] != signature:
            raise ValueError("Saved progress uses different settings. Choose another --out folder.")
        records = progress["records"]
        print("Resuming", len(records), "completed arm/seed/fold runs")
    else:
        records = []

    
    
    
    prior_path = args.previous_search
    if not prior_path.exists() and Path("question2_search_output.zip").exists():
        prior_path = Path("question2_search_output.zip")
    if not args.no_reuse and prior_path.exists():
        previous = load_previous(prior_path)
        ps = previous.get("signature", {})
        compatible = (ps.get("seeds") == args.seeds
                      and ps.get("folds") == signature["folds"]
                      and all(ps.get(k) == signature[k] for k in
                              ("epochs", "patience", "batch_size", "stride",
                               "width", "kernel", "top_k")))
        if compatible:
            reused = 0
            for prior in previous.get("records", []):
                if (prior.get("candidate") != "hybrid_672"
                        or prior.get("context") != CONTEXT
                        or prior.get("loss") != "hybrid_raw_log"):
                    continue
                pair = (True, prior["seed"], prior["fold_id"])
                if any((r["external"], r["seed"], r["fold_id"]) == pair
                       for r in records):
                    continue
                records.append({"external": True, "seed": prior["seed"],
                                "fold_id": prior["fold_id"],
                                "fit_end": prior["fit_end"],
                                "validation_origins": prior["validation_origins"],
                                "best_epoch": prior["best_epoch"],
                                "epochs_run": prior["epochs_run"],
                                "best_validation": prior["best_validation"],
                                "history": prior["history"],
                                "source": "compatible_prior_search"})
                reused += 1
            write_progress(progress_file, signature, records)
            print("Reused", reused, "compatible external-data runs")
        else:
            print("Prior search settings differ; both arms will be trained.")

    for fold_id, (fit_end, val_end) in enumerate(folds):
        val_origins = np.arange(fit_end, val_end-HORIZON+1, 672)
        for seed in args.seeds:
            for use_external in (True, False):
                pair = (use_external, seed, fold_id)
                if any((r["external"], r["seed"], r["fold_id"]) == pair
                       for r in records):
                    continue
                _, _, history, best_epoch = train_run(
                    y, external, fit_end, val_origins, seed,
                    use_external, args.epochs, args.batch_size,
                    args.stride, device, args.width, args.kernel,
                    args.top_k, args.patience, loss_name="hybrid_raw_log")
                records.append({"external": use_external, "seed": seed,
                                "fold_id": fold_id, "fit_end": fit_end,
                                "validation_origins": val_origins.tolist(),
                                "best_epoch": best_epoch,
                                "epochs_run": len(history),
                                "best_validation": min(history,
                                                       key=lambda row: row["rmse"]),
                                "history": history, "source": "new_training"})
                write_progress(progress_file, signature, records)

    assert len(records) == 2*len(args.seeds)*len(folds)
    rows = []
    for fold_id, (fit_end, _) in enumerate(folds):
        for seed in args.seeds:
            with_cov = next(r for r in records if r["external"] and
                            r["fold_id"] == fold_id and r["seed"] == seed)
            without_cov = next(r for r in records if not r["external"] and
                               r["fold_id"] == fold_id and r["seed"] == seed)
            assert with_cov["validation_origins"] == without_cov["validation_origins"]
            a, b = without_cov["best_validation"], with_cov["best_validation"]
            row = {"fit_end": fit_end, "fold": fold_id, "seed": seed,
                   "without_rmse": a["rmse"], "with_rmse": b["rmse"],
                   "rmse_improvement": a["rmse"]-b["rmse"],
                   "without_mae": a["mae"], "with_mae": b["mae"],
                   "without_smape": a["smape"], "with_smape": b["smape"],
                   "without_best_epoch": without_cov["best_epoch"],
                   "with_best_epoch": with_cov["best_epoch"]}
            rows.append(row)
    pd.DataFrame(rows).to_csv(args.out / "report_ablation_table.csv", index=False)
    summary = {"architecture": "Autoformer with series decomposition and Auto-Correlation",
               "context": CONTEXT, "horizon": HORIZON,
               "loss": "0.7 raw-scale MSE (divided by 100^2) + 0.3 standardized log MSE",
               "folds": signature["folds"], "seeds": args.seeds,
               "n_val_blocks_by_fold": [len(np.arange(f, z-HORIZON+1, 672))
                                        for f, z in folds],
               "metrics": {}, "paired_rmse_differences": rows}
    for field in ("rmse", "mae", "smape"):
        summary["metrics"][field] = {
            "without_mean": float(np.mean([r[f"without_{field}"] for r in rows])),
            "with_mean": float(np.mean([r[f"with_{field}"] for r in rows])),
            "mean_paired_improvement": float(np.mean(
                [r[f"without_{field}"]-r[f"with_{field}"] for r in rows])),
            "improvement_by_fold": [float(np.mean([
                r[f"without_{field}"]-r[f"with_{field}"]
                for r in rows if r["fold"] == fold_id]))
                for fold_id in range(len(folds))],
        }
    (args.out / "ablation_summary.json").write_text(json.dumps(summary, indent=2))
    print("Paired 672-step ablation:")
    print(pd.DataFrame(rows).to_string(index=False))
    print("Summary:", json.dumps(summary["metrics"], indent=2))
    print("Saved:", args.out / "ablation_summary.json")


if __name__ == "__main__":
    main()
