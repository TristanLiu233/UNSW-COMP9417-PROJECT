from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--start',default='2015-02-06')
    p.add_argument('--end',default='2015-02-12')
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    p.add_argument('--out', type=Path, default=script_dir)
    a=p.parse_args(); out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    rn='ridge_demand/outputs/tune/hdd17_cdd195/by_slot/ridge_predictions_by_slot_hdd17_cdd195_tuned.csv'
    hn='hgbr_demand/results/stage2/half_hour_predictions.csv.gz'
    ridge_path = project_dir / rn
    hgbr_path = project_dir / hn
    for path in (ridge_path, hgbr_path):
        if not path.is_file():
            raise FileNotFoundError(f'Required prediction file not found: {path}')
    r = pd.read_csv(ridge_path, parse_dates=['timestamp'])
    h = pd.read_csv(hgbr_path, compression='gzip', parse_dates=['timestamp'])
    assert r.timestamp.is_unique and h.timestamp.is_unique
    r=r.set_index('timestamp').sort_index();h=h.set_index('timestamp').sort_index()
    assert r.index.equals(h.index), 'Full replay timestamps differ'
    d=h[['actual_demand_mw','frozen']].rename(columns={'frozen':'hgbr_mw'}).join(r.rename(columns={'predicted_demand_mw':'ridge_mw'}),validate='one_to_one')
    assert np.isfinite(d.to_numpy()).all()
    day=(d.index-pd.Timedelta(minutes=30)).normalize()
    w=d.loc[(day>=pd.Timestamp(a.start))&(day<=pd.Timestamp(a.end))].copy()
    expected=pd.date_range(pd.Timestamp(a.start)+pd.Timedelta(minutes=30),pd.Timestamp(a.end)+pd.Timedelta(days=1),freq='30min')
    assert w.index.equals(expected), 'Window is incomplete'
    w.to_csv(out/'demand_comparison_window.csv')
    rows=[]
    for model,col in [('Ridge','ridge_mw'),('HGBR','hgbr_mw')]:
        e=(w[col]-w.actual_demand_mw).abs()
        rows.append({'model':model,'n_slots':len(w),'MAPE_pct':100*(e/w.actual_demand_mw).mean(),'MAE_MW':e.mean()})
    pd.DataFrame(rows).to_csv(out/'window_metrics.csv',index=False)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(11,3.1),layout='constrained')
    ax.plot(w.index,w.actual_demand_mw,color='black',lw=1.6,label='Observed',zorder=3)
    ax.plot(w.index,w.ridge_mw,color='#0072B2',lw=1.2,ls='--',label='Ridge')
    ax.plot(w.index,w.hgbr_mw,color='#D55E00',lw=1.2,label='HGBR')
    ax.set_ylabel('NSW demand (MW)');ax.set_xlabel('Interval-ending time (AEST)')
    ax.xaxis.set_major_locator(mdates.DayLocator());ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax.set_xlim(expected[0]-pd.Timedelta(minutes=30),expected[-1]);ax.grid(alpha=.18)
    ax.legend(ncol=3,loc='upper left',frameon=False)
    fig.savefig(out/'demand_observed_ridge_hgbr.png',dpi=300)
    fig.savefig(out/'demand_observed_ridge_hgbr.pdf')
    print(pd.DataFrame(rows).to_string(index=False));print('Full timestamp alignment verified:',len(d),'rows; window:',len(w),'rows')
if __name__=='__main__':main()
