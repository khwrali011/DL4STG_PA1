from __future__ import annotations
import argparse, copy, hashlib, json, math, random, time
from dataclasses import dataclass, asdict, replace
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
H=168
VERSION='attempt3-v1'


def save_json(p,obj):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,indent=2,allow_nan=False));t.replace(p)


def save_torch(p,obj):
    t=p.with_suffix('.tmp');torch.save(obj,t);t.replace(p)


def load_torch(p,device='cpu'):
    
    return torch.load(p,map_location=device,weights_only=False)


def seed_all(s):
    random.seed(s);np.random.seed(s);torch.manual_seed(s)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(s)


def metrics(y,p):
    y=np.asarray(y,dtype=np.float64);p=np.asarray(p,dtype=np.float64)
    if y.shape!=p.shape or not np.isfinite(p).all():raise ValueError('Invalid predictions')
    d=y-p
    return dict(rmse=float(np.sqrt(np.mean(d*d))),mae=float(np.mean(abs(d))),
                smape=float(np.mean(200*abs(d)/np.maximum(abs(y)+abs(p),1e-8))),
                mean_error=float(np.mean(p-y)))


def load_data(folder):
    a=pd.read_csv(folder/'student_train.csv');b=pd.read_csv(folder/'student_test.csv')
    e=pd.read_csv(folder/'optional_external_data.csv')
    assert list(a.columns)==['time_idx','value'] and list(b.columns)==['time_idx','value']
    assert list(e.columns)==['time_idx']+['feature_'+c for c in 'ABCDEFGHIJ']
    assert np.array_equal(a.time_idx,np.arange(1,43657))
    assert np.array_equal(b.time_idx,np.arange(43657,43825))
    assert np.array_equal(e.time_idx,np.arange(1,43825))
    assert b.value.isna().all(),'Test target must remain blank'
    y=a.value.to_numpy(dtype=np.float32);x=e.iloc[:,1:].to_numpy(dtype=np.float32)
    assert np.isfinite(y).all() and (y>=0).all() and np.isfinite(x).all()
    assert np.isin(x[:,6:], [0,1]).all() and (x[:,6:].sum(1)<=1).all()
    return y,x,b


@dataclass(frozen=True)
class Config:
    context:int=672
    width:int=48
    enc_layers:int=2
    kernel:int=25
    factor:float=1.0
    corr_normalized:bool=True
    target:str='raw'
    loss:str='raw'
    span:int=0
    half_life:int=0
    lr:float=0.0005
    dropout:float=0.1
    cov_head:bool=True
    cov_mode:str='all'  
    name:str=''


class Scale:
    def __init__(self,y,x,fit_end,cfg):
        start=max(0,fit_end-cfg.span) if cfg.span else 0
        a=y[start:fit_end].astype(np.float64)
        if cfg.target=='log':a=np.log1p(a)
        self.mode=cfg.target;self.mean=float(a.mean());self.std=float(max(a.std(),1.0 if cfg.target=='raw' else 1e-4))
        xx=self.external_base(x)
        self.em=xx[start:fit_end].mean(0);self.es=np.maximum(xx[start:fit_end].std(0),1e-4)
        self.em[6:]=0;self.es[6:]=1
        self.columns=list(range(10)) if cfg.cov_mode!='core' else [0,1,2,3,6,7,8,9]
        if cfg.cov_mode=='none':self.columns=[]
    @staticmethod
    def external_base(x):
        z=x.astype(np.float64).copy();z[:,3:6]=np.log1p(np.maximum(z[:,3:6],0));return z
    def transform(self,y,x):
        a=y.astype(np.float64)
        if self.mode=='log':a=np.log1p(a)
        return ((a-self.mean)/self.std).astype(np.float32),((self.external_base(x)-self.em)/self.es)[:,self.columns].astype(np.float32)
    def inverse(self,z):
        a=z*self.std+self.mean
        if self.mode=='log':a=np.expm1(np.clip(a,-20,12))
        return np.maximum(a,0)
    def raw_torch(self,z):
        a=z*self.std+self.mean
        return torch.expm1(a.clamp(-15,12)) if self.mode=='log' else a


class Windows(Dataset):
    def __init__(self,yn,en,y,origins,cfg,fit_end):
        self.yn,self.en,self.y,self.origins,self.cfg,self.fit_end=yn,en,y,list(map(int,origins)),cfg,fit_end
    def __len__(self):return len(self.origins)
    def __getitem__(self,i):
        o=self.origins[i];c=self.cfg.context
        assert o>=c and o+H<=self.fit_end,'Training target crossed cutoff'
        cov=self.en[o-c:o+H].copy()
        if self.cfg.cov_mode=='history':cov[c:]=0
        w=1.0 if not self.cfg.half_life else 0.25+0.75*2**(-(self.fit_end-o-H)/self.cfg.half_life)
        return torch.from_numpy(self.yn[o-c:o]),torch.from_numpy(cov),torch.from_numpy(self.yn[o:o+H]),torch.from_numpy(self.y[o:o+H]),np.float32(w)


class Decomp(nn.Module):
    def __init__(self,k):super().__init__();self.k=k
    def forward(self,x):
        trend=F.avg_pool1d(F.pad(x.transpose(1,2),(self.k//2,self.k//2),mode='replicate'),self.k,1).transpose(1,2)
        return x-trend,trend


class SeasonalNorm(nn.Module):
    def __init__(self,d):super().__init__();self.norm=nn.LayerNorm(d)
    def forward(self,x):
        x=self.norm(x);return x-x.mean(1,keepdim=True)


class LagMix(nn.Module):
    def __init__(self,d,factor,normalized=True):
        super().__init__();self.d=d;self.factor=factor;self.heads=4;self.normalized=normalized
        self.q=nn.Linear(d,d);self.k=nn.Linear(d,d);self.v=nn.Linear(d,d);self.out=nn.Linear(d,d)
    def forward(self,q,kv):
        b,l,d=q.shape
        if kv.shape[1]<l:kv=F.pad(kv.transpose(1,2),(0,l-kv.shape[1])).transpose(1,2)
        else:kv=kv[:,-l:]
        qq=self.q(q).view(b,l,4,d//4);kk=self.k(kv).view(b,l,4,d//4)
        vv=self.v(kv).transpose(1,2)
        corr=torch.fft.irfft(torch.fft.rfft(qq.float(),dim=1)*torch.fft.rfft(kk.float(),dim=1).conj(),n=l,dim=1).mean((2,3))
        if self.normalized:corr=corr/(l*math.sqrt(d//4))
        k=max(1,min(l,int(self.factor*math.log(l))))
        if self.training:
            delays=corr.mean(0).topk(k).indices[None].expand(b,-1);weights=corr.gather(1,delays).softmax(1)
        else:weights,delays=corr.topk(k,dim=1);weights=weights.softmax(1)
        pos=(torch.arange(l,device=q.device)[None,None,:]+delays[:,:,None])%l
        picked=vv[:,None].expand(-1,k,-1,-1).gather(3,pos[:,:,None,:].expand(-1,-1,d,-1))
        return self.out((picked*weights[:,:,None,None]).sum(1).transpose(1,2))


class EncoderBlock(nn.Module):
    def __init__(self,c):
        super().__init__();d=c.width;self.mix=LagMix(d,c.factor,c.corr_normalized);self.dec=Decomp(c.kernel);self.drop=nn.Dropout(c.dropout)
        self.ff=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Dropout(c.dropout),nn.Linear(4*d,d))
    def forward(self,x):
        x,_=self.dec(x+self.drop(self.mix(x,x)));x,_=self.dec(x+self.drop(self.ff(x)));return x


class DecoderBlock(nn.Module):
    def __init__(self,c):
        super().__init__();d=c.width;self.selfmix=LagMix(d,c.factor,c.corr_normalized);self.cross=LagMix(d,c.factor,c.corr_normalized)
        self.dec=Decomp(c.kernel);self.drop=nn.Dropout(c.dropout)
        self.ff=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Dropout(c.dropout),nn.Linear(4*d,d))
        self.trend=nn.Conv1d(d,1,3,padding=1,padding_mode='circular',bias=False)
    def forward(self,x,m):
        x,t1=self.dec(x+self.drop(self.selfmix(x,x)))
        x,t2=self.dec(x+self.drop(self.cross(x,m)))
        x,t3=self.dec(x+self.drop(self.ff(x)))
        return x,self.trend((t1+t2+t3).transpose(1,2)).squeeze(1)


class Autoformer(nn.Module):
    def __init__(self,c,n_ext):
        super().__init__();self.c=c;self.decomp=Decomp(c.kernel);d=c.width
        self.enc_embed=nn.Conv1d(1+n_ext,d,3,padding=1,padding_mode='circular')
        self.dec_embed=nn.Conv1d(1+n_ext,d,3,padding=1,padding_mode='circular')
        self.enc=nn.ModuleList([EncoderBlock(c) for _ in range(c.enc_layers)])
        self.dec=DecoderBlock(c);self.enorm=SeasonalNorm(d);self.dnorm=SeasonalNorm(d);self.out=nn.Linear(d,1)
        self.dropout=nn.Dropout(c.dropout)
        self.cov_head=None
        if c.cov_head and n_ext:
            self.cov_head=nn.Sequential(nn.Linear(n_ext+4,d),nn.GELU(),nn.Linear(d,1))
            nn.init.zeros_(self.cov_head[-1].weight);nn.init.zeros_(self.cov_head[-1].bias)
    def forward(self,h,e):
        c=self.c.context;assert h.shape[1]==c and e.shape[1]==c+H
        seasonal,trend=self.decomp(h.unsqueeze(-1))
        mem=self.dropout(self.enc_embed(torch.cat([h.unsqueeze(-1),e[:,:c]],-1).transpose(1,2)).transpose(1,2))
        for layer in self.enc:mem=layer(mem)
        mem=self.enorm(mem)
        s=torch.cat([seasonal[:,-H:],h.new_zeros((len(h),H,1))],1)
        t=torch.cat([trend[:,-H:],h.mean(1)[:,None,None].expand(-1,H,1)],1).squeeze(-1)
        d=self.dropout(self.dec_embed(torch.cat([s,e[:,c-H:]],-1).transpose(1,2)).transpose(1,2))
        d,dt=self.dec(d,mem);p=t+dt+self.out(self.dnorm(d)).squeeze(-1);p=p[:,-H:]
        if self.cov_head is not None:
            stats=torch.stack([h[:,-1],h.mean(1),h.std(1,unbiased=False),h[:,-H:].mean(1)],1)
            p=p+self.cov_head(torch.cat([e[:,-H:],stats[:,None,:].expand(-1,H,-1)],-1)).squeeze(-1)
        return p


def folds_for(n):
    folds=[]
    for offset in [80,48,24]:
        fit=n-offset*H;tune=list(range(fit,fit+4*H,H));dev=list(range(fit+4*H,fit+16*H,H))
        folds.append({'fit':fit,'tune':tune,'dev':dev})
    audit=list(range(n-8*H,n,H))
    assert all(o+H<=n-8*H for f in folds for o in f['dev'])
    assert len(set(o for f in folds for o in f['dev']))==36
    return folds,audit


@torch.no_grad()
def predict(model,scaler,y,x,origins,cfg,device):
    yn,en=scaler.transform(y,x);model.eval();preds=[]
    for o in origins:
        h=torch.from_numpy(yn[o-cfg.context:o]).unsqueeze(0).to(device)
        cov=en[o-cfg.context:o+H].copy()
        if cfg.cov_mode=='history':cov[cfg.context:]=0
        e=torch.from_numpy(cov).unsqueeze(0).to(device)
        preds.append(scaler.inverse(model(h,e).cpu().numpy()[0]))
    p=np.asarray(preds,dtype=np.float64)
    if p.shape!=(len(origins),H) or not np.isfinite(p).all():raise RuntimeError('Invalid forecast')
    return p


class Runner:
    def __init__(self,y,x,args,device):
        self.y,self.x,self.args,self.device=y,x,args,device
        self.cache=args.out/'cache';self.cache.mkdir(parents=True,exist_ok=True)
        self.generator=torch.Generator()
    def ledger(self):
        records=[json.loads(p.read_text()) for p in self.cache.glob('*.json')]
        
        return records
    def seconds(self):return sum(r['fit_seconds'] for r in self.ledger())
    def run(self,cfg,seed,stage,fit,tune,origins,epochs,patience=None):
        key=hashlib.sha256(json.dumps([asdict(cfg),seed,stage,fit,tune,origins,epochs],sort_keys=True).encode()).hexdigest()[:18]
        folder=self.cache/key;folder.mkdir(exist_ok=True);meta=self.cache/(key+'.json');cp=folder/'resume.pt';done=folder/'model.pt'
        if meta.exists() and done.exists():
            r=json.loads(meta.read_text());return r,np.load(folder/'predictions.npy')
        seed_all(seed);scale=Scale(self.y,self.x,fit,cfg);yn,en=scale.transform(self.y,self.x)
        first=cfg.context+max(0,fit-cfg.span) if cfg.span else cfg.context
        train_origins=np.arange(first,fit-H+1,self.args.stride)
        if self.args.smoke:train_origins=train_origins[:2*self.args.batch_size]
        if not len(train_origins):raise ValueError('No training windows')
        data=Windows(yn,en,self.y,train_origins,cfg,fit)
        gen=torch.Generator().manual_seed(seed)
        loader=DataLoader(data,batch_size=self.args.batch_size,shuffle=True,generator=gen,num_workers=0,pin_memory=self.device.type=='cuda')
        model=Autoformer(cfg,en.shape[1]).to(self.device)
        opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=1e-4)
        history=[];best=float('inf');best_epoch=0;stale=0;best_state=None;fit_seconds=0.0
        if cp.exists():
            state=load_torch(cp);model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
            for st in opt.state.values():
                for k,v in st.items():
                    if torch.is_tensor(v):st[k]=v.to(self.device)
            history=state['history'];best=state['best'];best_epoch=state['best_epoch'];stale=state['stale'];best_state=state['best_state'];fit_seconds=state['fit_seconds']
            torch.set_rng_state(state['torch_rng']);np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng']);gen.set_state(state['loader_rng'])
            if self.device.type=='cuda' and state['cuda_rng'] is not None:torch.cuda.set_rng_state_all(state['cuda_rng'])
            print('RESUME',stage,cfg.name,seed,'after epoch',len(history),flush=True)
        for epoch in range(len(history)+1,epochs+1):
            if tune and stale>=patience and len(history)>=self.args.min_epochs:break
            began=time.monotonic();model.train();losses=[]
            
            for h,e,z,raw,w in loader:
                h,e,z,raw,w=[a.to(self.device) for a in (h,e,z,raw,w)]
                opt.zero_grad(set_to_none=True);p=model(h,e)
                raw_p=scale.raw_torch(p)
                raw_loss=((raw_p-raw)/100).square().mean(1)
                loss=((p-z).square().mean(1)) if cfg.loss=='log' else raw_loss
                if cfg.loss=='mixed':loss=0.7*raw_loss+0.3*(p-z).square().mean(1)
                loss=(loss*w).sum()/w.sum()
                if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss in {cfg.name}')
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();losses.append(float(loss.detach()))
            rec={'epoch':epoch,'loss':float(np.mean(losses))}
            if tune:
                pp=predict(model,scale,self.y,self.x,tune,cfg,self.device)
                yy=np.stack([self.y[o:o+H] for o in tune]);rec.update(metrics(yy,pp))
                if rec['rmse']<best:
                    best=rec['rmse'];best_epoch=epoch;stale=0;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                else:stale+=1
            else:best_epoch=epoch
            if self.device.type=='cuda':torch.cuda.synchronize()
            fit_seconds+=time.monotonic()-began;history.append(rec)
            save_torch(cp,{'model':model.state_dict(),'optimizer':opt.state_dict(),'history':history,'best':best,'best_epoch':best_epoch,'stale':stale,'best_state':best_state,'fit_seconds':fit_seconds,'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),'loader_rng':gen.get_state(),'cuda_rng':torch.cuda.get_rng_state_all() if self.device.type=='cuda' else None})
            print(f'{stage} {cfg.name} seed={seed} epoch={epoch}/{epochs} loss={rec["loss"]:.4f} tune_RMSE={rec.get("rmse",0):.3f} completed_train_min={self.seconds()/60+fit_seconds/60:.1f}',flush=True)
        if tune:model.load_state_dict(best_state)
        pp=predict(model,scale,self.y,self.x,origins,cfg,self.device)
        r={'key':key,'cfg':asdict(cfg),'seed':seed,'stage':stage,'fit_end':fit,'epochs_executed':len(history),'best_epoch':best_epoch,'P':sum(p.numel() for p in model.parameters() if p.requires_grad),'fit_seconds':fit_seconds,'history':history,'checkpoint':str(done)}
        if max(origins)<len(self.y):r['metrics']=metrics(np.stack([self.y[o:o+H] for o in origins]),pp)
        save_torch(done,{'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'config':asdict(cfg),'scaler':vars(scale),'seed':seed})
        np.save(folder/'predictions.npy',pp);save_json(meta,r)
        del model,opt
        if self.device.type=='cuda':torch.cuda.empty_cache()
        return r,pp


def configuration_pool():
    base=Config()
    variants=[base,replace(base,target='log',loss='mixed'),replace(base,width=32,context=336,enc_layers=1),
              replace(base,width=64),replace(base,context=1008,kernel=49),replace(base,kernel=7),
              replace(base,corr_normalized=False),replace(base,cov_mode='core'),replace(base,cov_head=False),replace(base,span=24000),
              replace(base,half_life=8000),replace(base,target='log',loss='raw'),replace(base,factor=2),
              replace(base,context=336,width=48),replace(base,kernel=49),replace(base,lr=0.0002),
              replace(base,enc_layers=1,dropout=0.2)]
    rng=random.Random(20261004)
    for _ in range(96):
        variants.append(Config(context=rng.choice([336,672,1008]),width=rng.choice([32,48,64]),
            enc_layers=rng.choice([1,2]),kernel=rng.choice([7,25,49]),factor=rng.choice([1.0,2.0]),
            target=rng.choice(['raw','log']),span=rng.choice([0,24000]),half_life=rng.choice([0,8000]),
            lr=rng.choice([0.0002,0.0005]),cov_head=rng.choice([True,False]),cov_mode=rng.choice(['all','core'])))
    pool=[];seen=set()
    for c in variants:
        if c.target=='log' and c.loss=='raw':c=replace(c,loss='mixed') if len(pool)>16 else c
        key=json.dumps(asdict(c),sort_keys=True)
        if key in seen:continue
        seen.add(key);pool.append(replace(c,name=f'cfg{len(pool):03d}_{c.target}_{c.context}_w{c.width}'))
    return pool


def score_predictions(actual,pred,fold_sizes):
    m=metrics(actual,pred);per=[];i=0
    for size in fold_sizes:per.append(metrics(actual[i:i+size],pred[i:i+size])['rmse']);i+=size
    block=np.sqrt(np.mean((actual-pred)**2,axis=1))
    m.update(fold_rmse=per,block_rmse=block.tolist(),p90_block_rmse=float(np.percentile(block,90)))
    
    m['selection_score']=0.75*m['rmse']+0.25*max(per)
    return m


def calibration_fit(p,y):
    
    x=p.ravel()/100;z=y.ravel()/100;design=np.stack([x,np.ones_like(x)],1)
    penalty=0.1*len(x);A=design.T@design+penalty*np.eye(2)
    b=design.T@z+penalty*np.array([1.,0.]);a,b=np.linalg.solve(A,b)
    return float(np.clip(a,0.7,1.5)),float(np.clip(100*b,-30,30))


def calibration_crosscheck(p,y,sizes):
    
    
    calibrated=p.copy();i=sizes[0]
    for size in sizes[1:]:
        a,b=calibration_fit(p[:i],y[:i]);calibrated[i:i+size]=np.maximum(a*p[i:i+size]+b,0);i+=size
    early=sizes[0]
    old=metrics(y[early:],p[early:])['rmse'];new=metrics(y[early:],calibrated[early:])['rmse']
    ok=new<=0.98*old
    i=early
    for size in sizes[1:]:
        ok=ok and metrics(y[i:i+size],calibrated[i:i+size])['rmse']<=1.03*metrics(y[i:i+size],p[i:i+size])['rmse'];i+=size
    return bool(ok),dict(prequential_raw_rmse=old,prequential_calibrated_rmse=new,
                         fit_on_all_development=calibration_fit(p,y))


def audit_diagnostics(y,x,origins,p):
    actual=np.stack([y[o:o+H] for o in origins]);r=metrics(actual,p)
    r['block_rmse']=np.sqrt(np.mean((actual-p)**2,axis=1)).tolist()
    r['horizon_bins']={f'{a+1}-{b}':metrics(actual[:,a:b],p[:,a:b]) for a,b in [(0,24),(24,72),(72,168)]}
    high=actual>=np.percentile(y[:origins[0]],90)
    r['high_target_rmse']=float(np.sqrt(np.mean((actual[high]-p[high])**2))) if high.any() else None
    return r


def plot_results(out,y,origins,p,test_pred,ranking):
    try:
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib unavailable; JSON/CSV outputs still complete.');return
    fig,axs=plt.subplots(4,2,figsize=(13,12))
    for ax,o,pp in zip(axs.ravel(),origins,p):
        ax.plot(np.arange(H),y[o:o+H],label='observed');ax.plot(pp,label='frozen pipeline')
        ax.set_title(f'Audit indices {o+1}–{o+H}; RMSE {metrics(y[o:o+H],pp)["rmse"]:.2f}');ax.grid(alpha=.2)
    axs.ravel()[0].legend();fig.tight_layout();fig.savefig(out/'audit_forecasts.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,4));ax.plot(np.arange(len(y)-672,len(y)),y[-672:],label='observed target')
    ax.plot(np.arange(len(y),len(y)+H),test_pred,label='test forecast');ax.axvline(len(y),color='gray',linestyle='--')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'test_forecast.png',dpi=160);plt.close(fig)


def self_test():
    seed_all(1);torch.set_num_threads(2)
    for mode in ['raw','log']:
        for context in [336,672,1008]:
            cfg=Config(context=context,width=16,enc_layers=1,target=mode,cov_head=True)
            model=Autoformer(cfg,10);h=torch.randn(2,context);e=torch.randn(2,context+H,10)
            p=model(h,e);assert p.shape==(2,H) and torch.isfinite(p).all()
            p.square().mean().backward();assert all(torch.isfinite(t.grad).all() for t in model.parameters() if t.grad is not None)
            model.eval();assert torch.isfinite(model(h,e)).all()
    x=torch.randn(2,31,3);s,t=Decomp(7)(x);assert torch.allclose(s+t,x,atol=1e-6)
    
    y=np.arange(1000,dtype=np.float32);e=np.zeros((1168,10),dtype=np.float32)
    for mode in ['raw','log']:
        cfg=Config(target=mode);a=Scale(y,e,800,cfg);altered=y.copy();altered[800:]=100000
        b=Scale(altered,e,800,cfg);assert a.mean==b.mean and a.std==b.std
        z,_=a.transform(y,e);assert np.allclose(a.inverse(z),y,atol=.001)
    folds,audit=folds_for(43656);assert audit[0]==42312 and audit[-1]+H==43656
    print('SELF-TEST PASSED: forward/backward, train/eval FFT paths, decomposition, scaler isolation, split bounds.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=Path('Data'));p.add_argument('--out',type=Path,default=Path('question2_comprehensive_output'))
    p.add_argument('--device',choices=['auto','cuda','cpu'],default='auto')
    p.add_argument('--seeds',type=int,nargs='+',default=[17,29,43])
    p.add_argument('--batch-size',type=int,default=16);p.add_argument('--stride',type=int,default=8)
    p.add_argument('--screen-configs',type=int,default=10);p.add_argument('--screen-epochs',type=int,default=6)
    p.add_argument('--refine-configs',type=int,default=3);p.add_argument('--epochs',type=int,default=16)
    p.add_argument('--patience',type=int,default=4);p.add_argument('--min-epochs',type=int,default=3)
    p.add_argument('--min-train-minutes',type=float,default=60);p.add_argument('--max-search-minutes',type=float,default=90)
    p.add_argument('--calibration',choices=['auto','off'],default='auto')
    p.add_argument('--self-test',action='store_true');p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    if args.self_test:self_test();return
    if len(set(args.seeds))<3 or len(args.seeds)!=len(set(args.seeds)):p.error('Use at least three distinct seeds')
    if min(args.batch_size,args.stride,args.epochs,args.screen_epochs,args.patience,args.min_epochs)<1:p.error('Positive training arguments required')
    if args.screen_configs<3 or args.refine_configs<1 or args.max_search_minutes<args.min_train_minutes:p.error('Invalid search limits')
    if args.min_epochs>min(args.epochs,args.screen_epochs):p.error('min-epochs must not exceed screen-epochs or epochs')
    if args.smoke:
        args.out=args.out.with_name(args.out.name+'_smoke');args.screen_configs=3;args.refine_configs=1
        args.epochs=args.screen_epochs=args.min_epochs=1;args.min_train_minutes=0;args.max_search_minutes=1
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
    if device.type=='cpu':torch.set_num_threads(min(8,torch.get_num_threads()))
    y,x,test=load_data(args.data_dir);args.out.mkdir(parents=True,exist_ok=True)
    folds,audit=folds_for(len(y));pool=configuration_pool()
    
    signature={k:v for k,v in vars(args).items() if k not in ['data_dir','out','device','min_train_minutes','max_search_minutes','self_test']}
    signature.update(code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),data_sha256=hashlib.sha256(y.tobytes()+x.tobytes()).hexdigest())
    settings=args.out/'settings.json'
    if settings.exists() and json.loads(settings.read_text())!=signature:raise RuntimeError('Changed data/code/settings; use a new --out folder')
    save_json(settings,signature)
    save_json(args.out/'split_protocol.json',{'indexing':'zero-based forecast origins; fit excludes cutoff','folds':folds,'audit_origins':audit,'audit_note':'Reserved within this new search; portions have appeared in previous experiments. Not a globally untouched dataset.'})
    runner=Runner(y,x,args,device);screened=[];refined={};development_actual=np.concatenate([np.stack([y[o:o+H] for o in f['dev']]) for f in folds])
    sizes=[len(f['dev']) for f in folds]
    print('Device:',device,'Actual completed training floor (minutes):',args.min_train_minutes,flush=True)

    def screen(cfg):
        
        pp=[];aa=[]
        for j in [0,2]:
            f=folds[j];r,pred=runner.run(cfg,args.seeds[0],f'screen_f{j}',f['fit'],f['tune'],f['dev'],args.screen_epochs,args.patience)
            pp.append(pred);aa.append(np.stack([y[o:o+H] for o in f['dev']]))
        result={'config':asdict(cfg),**score_predictions(np.concatenate(aa),np.concatenate(pp),[12,12])}
        screened.append(result);save_json(args.out/'screening.json',screened)
        print('SCREEN:',cfg.name,result['selection_score'],flush=True)
        return result

    def refine(cfg):
        if cfg.name in refined:return
        records={};predictions={}
        for seed in args.seeds:
            pp=[];rr=[]
            for j,f in enumerate(folds):
                r,pred=runner.run(cfg,seed,f'refine_f{j}',f['fit'],f['tune'],f['dev'],args.epochs,args.patience)
                pp.append(pred);rr.append(r)
            records[seed]=rr;predictions[seed]=np.concatenate(pp)
        result={'config':asdict(cfg),'seed_metrics':{str(s):score_predictions(development_actual,predictions[s],sizes) for s in args.seeds},
                'ensemble_metrics':score_predictions(development_actual,np.mean(list(predictions.values()),axis=0),sizes)}
        refined[cfg.name]={'cfg':cfg,'records':records,'predictions':predictions,'summary':result}
        save_json(args.out/'refined_summary.json',[v['summary'] for v in refined.values()])
        print('REFINED:',cfg.name,result['ensemble_metrics']['selection_score'],flush=True)

    for cfg in pool[:args.screen_configs]:
        if len(screened)>=3 and runner.seconds()/60>=args.max_search_minutes:break
        screen(cfg)
    shortlist=sorted(screened,key=lambda r:r['selection_score'])[:args.refine_configs]
    for result in shortlist:
        if refined and runner.seconds()/60>=args.max_search_minutes and runner.seconds()/60>=args.min_train_minutes:break
        refine(Config(**result['config']))
    
    
    
    next_idx=len(screened)
    while runner.seconds()/60<args.min_train_minutes:
        if next_idx>=len(pool):raise RuntimeError('Candidate pool exhausted before requested floor; lower --min-train-minutes or enlarge workload')
        cfg=pool[next_idx];next_idx+=1;screen(cfg);refine(cfg)
    
    
    print('Search floor met:',runner.seconds()/60,'minutes. Freezing selection.',flush=True)
    options=[]
    
    singleton_seed=43 if 43 in args.seeds else args.seeds[-1]
    for v in refined.values():
        for kind,seeds in [('single',[singleton_seed]),('seed_ensemble',args.seeds)]:
            pp=np.mean([v['predictions'][s] for s in seeds],axis=0)
            m=score_predictions(development_actual,pp,sizes)
            members=[{'config':asdict(v['cfg']),'seed':s,'weight':1/len(seeds)} for s in seeds]
            options.append({'name':v['cfg'].name+'_'+kind,'members':members,'metrics':m,'predictions':pp})
    
    leaders=sorted(refined.values(),key=lambda v:v['summary']['ensemble_metrics']['selection_score'])[:2]
    if len(leaders)==2:
        members=[{'config':asdict(v['cfg']),'seed':s,'weight':1/(2*len(args.seeds))} for v in leaders for s in args.seeds]
        pp=np.mean([v['predictions'][s] for v in leaders for s in args.seeds],axis=0)
        options.append({'name':'two_config_ensemble','members':members,'metrics':score_predictions(development_actual,pp,sizes),'predictions':pp})
    options.sort(key=lambda v:v['metrics']['selection_score'])
    
    near=[o for o in options if o['metrics']['selection_score']<=1.01*options[0]['metrics']['selection_score']]
    def option_parameters(o):
        return sum(refined[m['config']['name']]['records'][m['seed']][0]['P'] for m in o['members'])
    selected=min(near,key=lambda o:(option_parameters(o),o['metrics']['selection_score']))
    calibration=(1.,0.);calibration_report=None
    if args.calibration=='auto':
        allowed,calibration_report=calibration_crosscheck(selected['predictions'],development_actual,sizes)
        if allowed:calibration=tuple(calibration_report['fit_on_all_development'])
    frozen={'name':selected['name'],'members':selected['members'],'development':selected['metrics'],
            'affine_calibration':list(calibration),'calibration_check':calibration_report,
            'selection_rule':'0.75 pooled RMSE + 0.25 worst-fold RMSE; prefer fewer parameters within 1% of best',
            'epoch_rule':'median of best epochs across three development folds, per selected seed'}
    for m in frozen['members']:
        rr=refined[m['config']['name']]['records'][m['seed']]
        m['final_epochs']=max(1,int(round(np.median([r['best_epoch'] for r in rr]))))
    save_json(args.out/'frozen_selection.json',frozen)
    save_json(args.out/'candidate_ranking.json',[{'name':o['name'],'parameters':option_parameters(o),'metrics':o['metrics']} for o in options])
    if args.smoke:
        print('SMOKE COMPLETED. Training/search exercised; no leaderboard file exported.');return
    
    audit_pred=np.zeros((len(audit),H));test_pred=np.zeros(H);audit_records=[];final_records=[]
    for m in frozen['members']:
        cfg=Config(**m['config']);seed=m['seed'];ep=m['final_epochs']
        r,pp=runner.run(cfg,seed,'audit',audit[0],[],audit,ep)
        audit_records.append(r);audit_pred+=m['weight']*pp
        r,pp=runner.run(cfg,seed,'final',len(y),[],[len(y)],ep)
        final_records.append(r);test_pred+=m['weight']*pp[0]
    a,b=calibration;audit_pred=np.maximum(a*audit_pred+b,0);test_pred=np.maximum(a*test_pred+b,0)
    report={'selection':frozen,'audit':audit_diagnostics(y,x,audit,audit_pred),'audit_origins':audit,
            'limitations':'Audit is withheld from this search, but previous experiments examined parts of it. Actual hidden-test metrics remain unknown.',
            'baseline_audit':{
                'last_value':metrics(np.stack([y[o:o+H] for o in audit]),np.stack([np.repeat(y[o-1],H) for o in audit])),
                'repeat_last_168':metrics(np.stack([y[o:o+H] for o in audit]),np.stack([y[o-H:o] for o in audit]))}}
    save_json(args.out/'confirmation_report.json',report)
    
    main_cfg=Config(**frozen['members'][0]['config']);f=folds[-1];ablation=[]
    without=replace(main_cfg,cov_mode='none',cov_head=False,name=main_cfg.name+'_without_optional')
    for seed in args.seeds:
        with_r,_=runner.run(main_cfg,seed,'refine_f2',f['fit'],f['tune'],f['dev'],args.epochs,args.patience)
        without_r,_=runner.run(without,seed,'ablation_f2',f['fit'],f['tune'],f['dev'],args.epochs,args.patience)
        ablation.append({'seed':seed,'with_optional':with_r['metrics'],'without_optional':without_r['metrics']})
    save_json(args.out/'optional_ablation.json',{'configuration':asdict(main_cfg),'split':f,'runs':ablation})
    assert len(test_pred)==168 and np.isfinite(test_pred).all() and (test_pred>=0).all()
    pd.DataFrame({'time_idx':test.time_idx,'value':test_pred}).to_csv(args.out/'forecast.csv',index=False)
    (args.out/'leaderboard_values.txt').write_text(', '.join(f'{v:.6f}' for v in test_pred)+'\n')
    
    keys=set();lineage_epochs=0
    for m in frozen['members']:
        for r in refined[m['config']['name']]['records'][m['seed']]:
            if r['key'] not in keys:lineage_epochs+=r['epochs_executed'];keys.add(r['key'])
    for r in audit_records+final_records:
        if r['key'] not in keys:lineage_epochs+=r['epochs_executed'];keys.add(r['key'])
    P=sum(r['P'] for r in final_records)+(2 if calibration!=(1.,0.) else 0)
    declaration={'parameters_P_for_form':P,'epochs_E_selected_pipeline':lineage_epochs,
                 'final_fit_epochs_sum':sum(r['epochs_executed'] for r in final_records),
                 'search_and_ablation_epochs_total':sum(r['epochs_executed'] for r in runner.ledger()),
                 'epoch_explanation':'Selected three-fold early-stop runs + selected audit fits + selected final fits, summed across ensemble members; unrelated search and ablation excluded.',
                 'calibration_coefficients_counted_in_P':2 if calibration!=(1.,0.) else 0,
                 'ensemble_size':len(final_records),'members':frozen['members'],'actual_completed_fit_minutes':runner.seconds()/60,
                 'audit_metrics':report['audit'],'test_metrics':None,'test_indices':[43657,43824],
                 'final_checkpoints':[r['checkpoint'] for r in final_records]}
    save_json(args.out/'declaration.json',declaration)
    rows=[]
    for i,o in enumerate(audit):
        for j in range(H):rows.append({'time_idx':o+j+1,'origin':o+1,'horizon':j+1,'actual':float(y[o+j]),'prediction':float(audit_pred[i,j])})
    pd.DataFrame(rows).to_csv(args.out/'audit_predictions.csv',index=False)
    plot_results(args.out,y,audit,audit_pred,test_pred,options)
    print('\nDONE:',args.out/'leaderboard_values.txt','\nP:',P,'E selected pipeline:',lineage_epochs,
          '\nAudit RMSE:',report['audit']['rmse'],'\nSend confirmation_report.json, declaration.json and candidate_ranking.json for review.',flush=True)


if __name__=='__main__':main()
