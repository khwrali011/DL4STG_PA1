from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


HORIZON = 168
CONTEXT = 336
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
    assert x.shape[1] == 10, "Expected the ten assignment optional columns in original order"
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
              fixed_epochs=None, loss_name="log_mse", train_span=0):
    seed_all(seed)
    scaler = Scaler(y, ext, fit_end)
    yn, en = scaler.transform(y, ext)
    if not use_external:
        en = en[:, :0].copy()
    first_origin = max(CONTEXT, fit_end - train_span + CONTEXT) if train_span else CONTEXT
    origins = np.arange(first_origin, fit_end-HORIZON+1, stride)
    if not len(origins):
        raise ValueError("Not enough history for this training span")
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
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss; rerun in a new output folder with a different setting")
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



def atomic_json(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


@torch.no_grad()
def predict_origins(model, scaler, y, ext, origins, device):
    yn, en = scaler.transform(y, ext)
    model.eval()
    result = []
    for o in origins:
        h = torch.from_numpy(yn[o-CONTEXT:o]).unsqueeze(0).to(device)
        x = torch.from_numpy(en[o-CONTEXT:o+HORIZON]).unsqueeze(0).to(device)
        result.append(scaler.inverse(model(h, x).cpu().numpy()[0]))
    result = np.asarray(result, dtype=np.float64)
    if result.shape != (len(origins), HORIZON) or not np.isfinite(result).all():
        raise RuntimeError('Invalid forecast; refusing to export')
    return result


def recent_schedule(n):
    
    inner_end, development_end, confirmation_start = n-12*HORIZON, n-8*HORIZON, n-4*HORIZON
    return inner_end, development_end, confirmation_start


def main():
    import hashlib
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('Data'))
    parser.add_argument('--out', type=Path, default=Path('question2_recent_output'))
    parser.add_argument('--seeds', type=int, nargs='+', default=[17,29,43])
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--stride', type=int, default=16)
    parser.add_argument('--max-minutes', type=float, default=60)
    parser.add_argument('--device', choices=['auto','cuda','cpu'], default='auto')
    parser.add_argument('--smoke', action='store_true', help='Tiny pipeline check; never exports submission values')
    args = parser.parse_args()
    if len(set(args.seeds)) < 2 or len(args.seeds) != len(set(args.seeds)):
        parser.error('Use at least two distinct seeds')
    if min(args.epochs,args.patience,args.batch_size,args.stride) < 1 or args.max_minutes <= 0:
        parser.error('Training arguments must be positive')
    if args.smoke:
        args.epochs, args.stride = 1, 1024
        args.out = args.out.with_name(args.out.name+'_smoke')
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    if device.type=='cpu':
        torch.set_num_threads(min(8, torch.get_num_threads()))
    y, ext, test = load_data(args.data_dir)
    if not np.isfinite(y).all() or not np.isfinite(ext).all():
        raise ValueError('Nonfinite input data')
    n=len(y)
    inner_end, dev_start, confirm_start=recent_schedule(n)
    inner_origins=np.arange(inner_end,dev_start,HORIZON)
    dev_origins=np.arange(dev_start,confirm_start,HORIZON)
    confirm_origins=np.arange(confirm_start,n,HORIZON)
    configs=[
        {'name':'mixed_672_full','loss':'hybrid_raw_log','context':672,'span':0},
        {'name':'raw_672_full','loss':'raw_mse','context':672,'span':0},
        {'name':'mixed_672_recent','loss':'hybrid_raw_log','context':672,'span':16000},
        {'name':'raw_336_recent','loss':'raw_mse','context':336,'span':16000},
        {'name':'mixed_336_full','loss':'hybrid_raw_log','context':336,'span':0},
    ]
    out=args.out;out.mkdir(parents=True,exist_ok=True)
    cache=out/'cache';cache.mkdir(exist_ok=True)
    fingerprint=hashlib.sha256(y.tobytes()+ext.tobytes()+Path(__file__).read_bytes()).hexdigest()
    settings={'data_code_sha256':fingerprint,'seeds':args.seeds,'epochs':args.epochs,
              'patience':args.patience,'batch_size':args.batch_size,'stride':args.stride,
              'configs':configs,'smoke':args.smoke}
    manifest=out/'settings.json'
    if manifest.exists() and json.loads(manifest.read_text()) != settings:
        raise RuntimeError('Data/code/settings differ from cached run. Choose a new --out directory.')
    atomic_json(manifest,settings)
    started=time.monotonic()
    print('Device:',device,'Recent splits:',inner_end,dev_start,confirm_start,n,flush=True)

    def run(cfg, seed, stage, fit_end, origins, fixed_epochs=None, save_model=False):
        global CONTEXT
        CONTEXT=cfg['context']
        key=f"{cfg['name']}_seed{seed}_{stage}"
        meta=cache/(key+'.json');arrays=cache/(key+'.npz')
        checkpoint=cache/(key+'.pt')
        if meta.exists() and arrays.exists() and (not save_model or checkpoint.exists()):
            print('Reusing completed:',key,flush=True)
            return json.loads(meta.read_text()),np.load(arrays)['predictions']
        model,scaler,log,best_epoch=train_run(
            y,ext,fit_end,origins if fixed_epochs is None else None,seed,True,args.epochs,
            args.batch_size,args.stride,device,32,25,5,args.patience,
            fixed_epochs=fixed_epochs,loss_name=cfg['loss'],train_span=cfg['span'])
        pred=predict_origins(model,scaler,y,ext,origins,device)
        actual=np.stack([y[o:o+HORIZON] for o in origins]) if max(origins)<n else None
        record={'config':cfg,'seed':seed,'stage':stage,'fit_end':fit_end,
                'origins':list(map(int,origins)),'epochs_executed':len(log),
                'best_epoch':best_epoch,'parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
                'history':log}
        if actual is not None:
            record['metrics']=metrics(actual,pred)
            record['block_rmse']=np.sqrt(np.mean((actual-pred)**2,axis=1)).tolist()
        tmp=arrays.with_suffix('.tmp')
        with tmp.open('wb') as f:np.savez_compressed(f,predictions=pred)
        tmp.replace(arrays)
        if save_model:
            temp=checkpoint.with_suffix('.tmp')
            torch.save({'state_dict':model.cpu().state_dict(),'config':cfg,'seed':seed,
                        'scaler':vars(scaler),'context':CONTEXT,'horizon':HORIZON},temp)
            temp.replace(checkpoint)
        atomic_json(meta,record)
        del model
        if device.type=='cuda':torch.cuda.empty_cache()
        return record,pred

    summaries=[]
    for cfg in configs:
        complete=all((cache/f"{cfg['name']}_seed{s}_development.json").exists() for s in args.seeds)
        
        if not complete and len(summaries)>=2 and (time.monotonic()-started)/60>args.max_minutes:
            print('Soft search budget reached. Finishing with completed configurations.',flush=True)
            break
        seed_records=[]
        for seed in args.seeds:
            tune,_=run(cfg,seed,'early_stopping',inner_end,inner_origins)
            dev,_=run(cfg,seed,'development',dev_start,dev_origins,fixed_epochs=tune['best_epoch'])
            seed_records.append({'seed':seed,'rmse':dev['metrics']['rmse'],
                                 'block_rmse':dev['block_rmse'],'refit_epochs':tune['best_epoch']})
        summary={'config':cfg,'mean_seed_rmse':float(np.mean([r['rmse'] for r in seed_records])),
                 'seeds':seed_records}
        summaries.append(summary)
        atomic_json(out/'development_summary.json',summaries)
        print('Completed:',cfg['name'],'mean recent RMSE:',summary['mean_seed_rmse'],flush=True)
    winner=min(summaries,key=lambda r:r['mean_seed_rmse'])
    chosen=min(winner['seeds'],key=lambda r:r['rmse'])
    cfg=winner['config'];seed=chosen['seed'];epochs=chosen['refit_epochs']
    
    confirmation,_=run(cfg,seed,'confirmation',confirm_start,confirm_origins,fixed_epochs=epochs)
    baselines={}
    actual=np.stack([y[o:o+HORIZON] for o in confirm_origins])
    for label,pred in {
        'last_value':np.stack([np.repeat(y[o-1],HORIZON) for o in confirm_origins]),
        'repeat_last_168':np.stack([y[o-HORIZON:o] for o in confirm_origins]),
    }.items():baselines[label]=metrics(actual,pred)
    confirmation_report={'selected_config':cfg,'selected_seed':seed,
                          'development_mean_seed_rmse':winner['mean_seed_rmse'],
                          'untouched_confirmation':confirmation['metrics'],
                          'confirmation_block_rmse':confirmation['block_rmse'],
                          'diagnostic_baselines':baselines,
                          'note':'Confirmation is diagnostic; no model/seed/epoch selection uses these targets.'}
    atomic_json(out/'confirmation_report.json',confirmation_report)
    print('UNTOUCHED CONFIRMATION:',confirmation_report,flush=True)
    if args.smoke:
        print('Smoke run complete. No submission file generated. Run without --smoke.');return
    final,pred=run(cfg,seed,'final',n,np.asarray([n]),fixed_epochs=epochs,save_model=True)
    forecast=pred[0]
    if len(forecast)!=len(test) or (forecast<0).any():raise RuntimeError('Invalid final predictions')
    pd.DataFrame({'time_idx':test.time_idx,'value':forecast}).to_csv(out/'forecast.csv',index=False)
    (out/'leaderboard_values.txt').write_text(', '.join(f'{v:.6f}' for v in forecast)+'\n')
    executed=[json.loads(p.read_text()) for p in cache.glob('*.json')]
    total=sum(r['epochs_executed'] for r in executed)
    lineage=sum(r['epochs_executed'] for r in executed if r['config']['name']==cfg['name'] and r['seed']==seed)
    declaration={'trainable_parameters_P':final['parameters'],'final_refit_epochs':epochs,
                 'selected_pipeline_epochs_including_validation':lineage,
                 'all_experiments_epochs_including_search':total,
                 'conservative_E_if_form_counts_all_training':total,
                 'selected_candidate':cfg,'selected_seed':seed,'uses_optional_data':True,
                 'test_time_idx':[int(test.time_idx.iloc[0]),int(test.time_idx.iloc[-1])],
                 'development_mean_seed_rmse':winner['mean_seed_rmse'],
                 'confirmation_metrics':confirmation['metrics'],
                 'epoch_note':'Counts are actual epochs executed. Use the count matching the form definition; all-training count is the conservative upper bound.',
                 'test_metrics':None,'elapsed_minutes_this_invocation':(time.monotonic()-started)/60}
    atomic_json(out/'declaration.json',declaration)
    pd.DataFrame([{'candidate':r['config']['name'],'mean_seed_rmse':r['mean_seed_rmse']} for r in summaries]).to_csv(out/'development_summary.csv',index=False)
    print('\nDONE. Paste predictions from:',out/'leaderboard_values.txt',flush=True)
    print('Parameters P:',final['parameters'],'Final fit epochs:',epochs,
          'Selected pipeline epochs:',lineage,'All experiments epochs:',total,flush=True)
    print('Share confirmation_report.json and declaration.json before using another submission attempt.',flush=True)


if __name__=='__main__':
    main()
