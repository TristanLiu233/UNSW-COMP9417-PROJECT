"""Export P4 handoff rows, yearly class shares and a real 37-43-day frozen window."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .scoring import intervals


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',required=True,type=Path)
    parser.add_argument('--results',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Use a new output directory')
    data=pd.read_csv(args.data,usecols=['timestamp','nsw_price'],index_col='timestamp',parse_dates=True)
    expected=pd.date_range('2010-01-01 00:30','2026-11-01 00:00',freq='30min')
    if not data.index.equals(expected):raise ValueError('Unexpected course time axis')
    days=(data.index-pd.Timedelta(minutes=30)).normalize()
    previous=days.map(data.nsw_price.groupby(days).agg(lambda v:v.mean()).shift(1))
    valid=previous.notna()
    labels=pd.DataFrame({'year':days.year,'class_1':(data.nsw_price.to_numpy()>previous).astype(int)},index=data.index).loc[valid]
    yearly=labels.groupby('year').class_1.agg(n_slots='size',n_class1='sum')
    yearly['n_class0']=yearly.n_slots-yearly.n_class1
    yearly['class1_percent']=100*yearly.n_class1/yearly.n_slots
    yearly['class0_percent']=100-yearly.class1_percent
    args.output.mkdir(parents=True)
    yearly.to_csv(args.output/'yearly_class_share.csv')
    fig,ax=plt.subplots(figsize=(11,4.5),layout='constrained')
    ax.bar(yearly.index,yearly.class0_percent,label='Class 0',color='#a4bcc2')
    ax.bar(yearly.index,yearly.class1_percent,bottom=yearly.class0_percent,label='Class 1',color='#176B87')
    ax.set(ylabel='Observed class share (%)',xlabel='Market year',ylim=(0,100),
           title='NSW price direction | 2010 excludes Jan 1; 2026 ends Oct 31')
    ax.set_xticks(list(range(2010,2027,2)))
    ax.legend(ncol=2);fig.savefig(args.output/'yearly_class_share.png',dpi=150);plt.close(fig)
    selected=json.loads((args.results/'selection_2014.json').read_text())
    scores=pd.read_csv(args.results/'validation_2014.csv')
    row=scores.loc[scores.model==selected['model']].copy()
    row.insert(0,'owner','P4');row['feature_v1_alignment']='pending_P1_confirmation'
    row['train_end']='2013-12-31';row['validation_year']=2014
    row.to_csv(args.output/'p4_2014_shared_row.csv',index=False)
    config={'status':'P4_candidate_only_not_group_frozen_configuration',
            'model':selected,'feature_v1_alignment':'pending_P1_confirmation',
            'group_feature_change_approval':'not_verified',
            'training_cutoff':'2014-12-31','random_state':17,
            'artifact_dir':'logistic_price/results/frozen/artifacts/'+selected['model'],
            'note':'For P1 to merge after interface review. Does not replace group results/stage1_config.json.'}
    (args.output/'p4_stage1_candidate.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    pred=pd.read_csv(args.results/'replay_predictions.csv.gz',index_col='timestamp',parse_dates=True)
    market_days=(pred.index-pd.Timedelta(minutes=30)).normalize()
    training_cutoff=pd.Timestamp('2014-12-31')
    offset=(market_days-training_cutoff).days
    block=pred.loc[(offset>=37)&(offset<=43)]
    assert len(block)==7*48 and block.index[0]==pd.Timestamp('2015-02-06 00:30')
    names=[c for c in block if c!='truth' and not c.endswith('_probability')]
    rows,_=intervals(block.truth.to_numpy(),{n:block[n].to_numpy() for n in names})
    frame=pd.DataFrame(rows)
    frame.insert(0,'training_cutoff','2014-12-31')
    frame['first_market_day']='2015-02-06';frame['last_market_day']='2015-02-12'
    frame['model_age_days']='37-43';frame['independent_week_count']=1
    frame.to_csv(args.output/'age_37_43_frozen_window.csv',index=False)
    block.to_csv(args.output/'age_37_43_predictions.csv.gz',index_label='timestamp')
    print('PASS: 17 yearly class-share rows, P4 shared row/candidate config, one real 336-slot age-37-to-43 window.')


if __name__=='__main__': main()
