"""Render predetermined replay windows and class proportions; export handoff predictions."""
import argparse
from pathlib import Path
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def generate(output):
    selected=json.loads((output/'run.json').read_text())['selected']
    p=pd.read_csv(output/'replay_predictions.csv.gz',index_col='timestamp',parse_dates=True)
    names=[c for c in p if c not in ('truth',) and not c.endswith('_probability')]
    months=(p.index-pd.Timedelta(minutes=30)).strftime('%Y-%m')
    balance=pd.DataFrame({'truth_positive_percent':p.truth.groupby(months).mean()*100})
    for name in names:balance[name+'_positive_percent']=p[name].groupby(months).mean()*100
    balance.to_csv(output/'monthly_class_balance.csv',index_label='market_month')
    fig,ax=plt.subplots(figsize=(12,4.5),layout='constrained')
    for name,col in [('truth','truth_positive_percent'),(selected,selected+'_positive_percent'),('log22','log22_positive_percent')]:
        ax.plot(pd.to_datetime(balance.index),balance[col],label=name)
    ax.set(title='Frozen replay | actual and predicted positive-class proportions',ylabel='Class 1 share (%)',xlabel='Market month')
    ax.grid(alpha=.2);ax.legend();fig.savefig(output/'class_balance.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(13,7),layout='constrained')
    for ax,(start,end) in zip(axes,[('2015-01-01 00:30','2015-01-08 00:00'),('2026-10-25 00:30','2026-11-01 00:00')]):
        block=p.loc[start:end]
        ax.scatter(block.index,block.truth,s=8,color='#677786',alpha=.55,label='Observed class')
        ax.plot(block.index,block[selected+'_probability'],color='#176B87',linewidth=1,label='Predicted P(class 1)')
        cfg=json.loads((output/'selection_2014.json').read_text())
        ax.axhline(cfg['threshold'],color='#D48528',linestyle='--',label='Decision threshold')
        ax.set(title=f'{selected} | {start[:10]} to {end[:10]}',ylabel='Probability / observed class',ylim=(-.06,1.06))
        ax.legend(ncol=3,fontsize=8);ax.grid(alpha=.15)
    fig.savefig(output/'observed_vs_predicted.png',dpi=150);plt.close(fig)
    for name in [selected,'log22']:
        handoff=pd.DataFrame({'nsw_price_up':p[name], 'price_up_probability':p[name+'_probability'], 'actual_price_up':p.truth})
        handoff.to_csv(output/f'handoff_{name}.csv.gz',index_label='timestamp')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--results',type=Path,required=True)
    generate(parser.parse_args().results)
