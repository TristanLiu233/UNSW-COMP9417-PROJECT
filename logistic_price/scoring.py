"""Pooled classification metrics, per-class normal intervals and paired day bootstrap."""
import numpy as np
import pandas as pd


def counts(y, p):
    return np.stack([(y == 0), (y == 1), (y == 0) & (p == 0), (y == 1) & (p == 1)], axis=-1).astype(float)


def from_counts(c):
    n0, n1, t0, t1 = np.moveaxis(c, -1, 0)
    recall0 = np.divide(t0, n0, out=np.zeros_like(t0), where=n0 > 0)
    recall1 = np.divide(t1, n1, out=np.zeros_like(t1), where=n1 > 0)
    classes = (n0 > 0).astype(float) + (n1 > 0).astype(float)
    ba = (recall0 + recall1) / classes
    acc = (t0 + t1) / (n0 + n1)
    return 100 * ba, 100 * acc


def intervals(y, predictions, repetitions=2000, seed=17):
    if len(y) % 48:
        raise ValueError('Day bootstrap requires complete 48-slot days')
    names = list(predictions)
    daily = np.stack([counts(y, predictions[n]).reshape(-1, 48, 4).sum(axis=1) for n in names], axis=1)
    days = len(daily)
    rng = np.random.default_rng(seed)
    ba_samples, acc_samples = [], []
    for start in range(0, repetitions, 200):
        weights = rng.multinomial(days, np.full(days, 1 / days), size=min(200, repetitions-start))
        totals = (weights @ daily.reshape(days, -1)).reshape(-1, len(names), 4)
        ba, acc = from_counts(totals)
        ba_samples.append(ba); acc_samples.append(acc)
    ba_samples, acc_samples = np.concatenate(ba_samples), np.concatenate(acc_samples)
    rows = []
    for i, name in enumerate(names):
        total = daily[:, i].sum(axis=0)
        ba, acc = from_counts(total)
        n0, n1, t0, t1 = total
        recalls = [t/n for t,n in [(t0,n0),(t1,n1)] if n > 0]
        # Conditional-on-class-count independent Bernoulli approximation.
        variance = sum((t/n)*(1-t/n)/n for t,n in [(t0,n0),(t1,n1)] if n > 0) / len(recalls)**2
        half = 196 * np.sqrt(variance)
        acc_half = 196 * np.sqrt((acc/100)*(1-acc/100)/(n0+n1))
        blo, bhi = np.quantile(ba_samples[:, i], [.025, .975])
        alo, ahi = np.quantile(acc_samples[:, i], [.025, .975])
        rows.append({'model': name, 'n_slots': len(y), 'n_days': days, 'n0': int(n0), 'n1': int(n1),
                     'BalAcc': float(ba), 'Accuracy': float(acc),
                     'Recall0': 100*t0/n0 if n0 else np.nan, 'Recall1': 100*t1/n1 if n1 else np.nan,
                     'BalAcc_iid_low': max(0, ba-half), 'BalAcc_iid_high': min(100, ba+half),
                     'Accuracy_iid_low': max(0, acc-acc_half), 'Accuracy_iid_high': min(100, acc+acc_half),
                     'BalAcc_day_low': blo, 'BalAcc_day_high': bhi,
                     'Accuracy_day_low': alo, 'Accuracy_day_high': ahi})
    return rows, {name: ba_samples[:, i] for i,name in enumerate(names)}


def summarize(predictions, selected, output):
    # Group on market-day start, never on the interval-end date at midnight.
    days = (predictions.index - pd.Timedelta(minutes=30)).normalize()
    columns = [c for c in predictions if c not in ('truth',) and not c.endswith('_probability')]
    tables = {}
    for frequency in ('overall', 'yearly', 'monthly'):
        labels = np.full(len(days), '2015-2026') if frequency == 'overall' else days.strftime('%Y' if frequency == 'yearly' else '%Y-%m')
        records = []
        for label in np.unique(labels):
            block = predictions.loc[labels == label]
            rows, samples = intervals(block.truth.to_numpy(), {c: block[c].to_numpy() for c in columns})
            records.extend([{'period': label, **r} for r in rows])
            if frequency == 'overall':
                pairs = []
                for reference in columns:
                    if reference == selected:
                        continue
                    delta = samples[selected] - samples[reference]
                    points = {r['model']: r['BalAcc'] for r in rows}
                    low, high = np.quantile(delta, [.025,.975])
                    pairs.append({'candidate': selected, 'reference': reference,
                                  'delta_BalAcc_pp': points[selected]-points[reference],
                                  'day_low': low, 'day_high': high})
                pd.DataFrame(pairs).to_csv(output/'paired_comparisons.csv', index=False)
        tables[frequency] = pd.DataFrame(records)
        tables[frequency].to_csv(output/f'{frequency}.csv', index=False)
    return tables


def plots(tables, selected, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = [selected, 'previous_raw22', 'lag1', 'lag7']
    colors = ['#176B87', '#D48528', '#677786', '#A56BA8']
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), layout='constrained')
    for name, color in zip(names, colors):
        frame = tables['monthly'].query('model == @name')
        dates = pd.to_datetime(frame.period)
        axes[0].plot(dates, frame.BalAcc, label=name, color=color, linewidth=1.2)
        if name == selected:
            axes[0].fill_between(dates, frame.BalAcc_day_low, frame.BalAcc_day_high, color=color, alpha=.15)
        axes[1].plot(dates, frame.Accuracy, label=name, color=color, linewidth=1.2)
    for ax, label in zip(axes, ['Balanced accuracy (%)', 'Accuracy (%)']):
        ax.set_ylabel(label); ax.grid(alpha=.2); ax.legend(ncol=2, fontsize=9)
    axes[0].set_title('Frozen Logistic replay | monthly pooled scores | selected-model day-bootstrap 95% band')
    axes[1].set_xlabel('Market month (2026 ends in October)')
    fig.savefig(output/'monthly_metrics.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 4.8), layout='constrained')
    for name, color in zip(names, colors):
        frame = tables['yearly'].query('model == @name')
        ax.plot(frame.period.astype(int), frame.BalAcc, marker='o', label=name, color=color)
    ax.set(title='Frozen Logistic replay | annual pooled balanced accuracy', ylabel='Balanced accuracy (%)',
           xlabel='Market year (2026: January-October)')
    ax.grid(alpha=.2); ax.legend(ncol=2)
    fig.savefig(output/'yearly_balacc.png', dpi=160); plt.close(fig)

    # Required accuracy intervals for every frozen model and both official baselines.
    all_names = list(tables['monthly'].model.unique())
    fig, axes = plt.subplots((len(all_names)+1)//2, 2, figsize=(14, 3*((len(all_names)+1)//2)), layout='constrained', sharex=True)
    for ax, name in zip(axes.flat, all_names):
        frame = tables['monthly'].query('model == @name')
        dates = pd.to_datetime(frame.period)
        ax.fill_between(dates, frame.Accuracy_day_low, frame.Accuracy_day_high, color='#176B87', alpha=.2, label='Day bootstrap 95%')
        ax.fill_between(dates, frame.Accuracy_iid_low, frame.Accuracy_iid_high, color='#D48528', alpha=.45, label='Independent normal 95%')
        ax.plot(dates, frame.Accuracy, color='#16394A', linewidth=.8)
        ax.set_title(name); ax.set_ylabel('Accuracy (%)'); ax.grid(alpha=.15)
    for ax in list(axes.flat)[len(all_names):]: ax.set_visible(False)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('Frozen replay: monthly accuracy and two interval assumptions | 2026 ends in October')
    fig.savefig(output/'monthly_accuracy_intervals.png', dpi=150); plt.close(fig)
