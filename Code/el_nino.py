"""Reproduce winter classification, exact inference and Figure 8 from source data."""
from pathlib import Path
from hashlib import sha256
from fractions import Fraction
from math import comb
import runpy
import sys

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from scipy.stats.contingency import odds_ratio
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT/'Figures'
INDICES = ['NINO3', 'NINO34', 'NINO4']


def exact_test(table):
    a, b, c, d = map(int, table.ravel())
    total, central, later = a+b+c+d, a+c, a+b
    support = range(max(0, later-(total-central)), min(later, central)+1)
    probabilities = {x: Fraction(comb(central, x)*comb(total-central, later-x),
                                 comb(total, later)) for x in support}
    assert sum(probabilities.values()) == 1
    probability = sum(p for p in probabilities.values() if p <= probabilities[a])
    np.testing.assert_allclose(float(probability), fisher_exact(table).pvalue, atol=1e-15)
    rows = [dict(later_central=x, later_eastern=later-x, earlier_central=central-x,
                 earlier_eastern=total-central-later+x, probability=float(p),
                 probability_fraction=str(p), included_in_two_sided_p=p <= probabilities[a])
            for x, p in probabilities.items()]
    return float(probability), pd.DataFrame(rows)


def count_table(frame):
    return np.array([[int(((frame.winter_year >= 2005) & (frame.winter_type == t)).sum())
                      for t in ['CP', 'EP']],
                     [int(((frame.winter_year <= 2004) & (frame.winter_type == t)).sum())
                      for t in ['CP', 'EP']]])


def calculate(winters=None):
    source = ROOT/'Data/Climate_Indices_Data.xlsx'
    if winters is None:
        winters = runpy.run_path(str(ROOT/'Code/winter_aggregation.py'))['calculate']()[0]
    w = winters.loc[1982:2026, INDICES]
    assert w.shape == (45, 3) and np.isfinite(w).all().all()
    z = (w-w.mean())/w.std(ddof=1)
    contrast = z.NINO4-z.NINO3
    catalogue = pd.DataFrame(dict(winter_year=w.index,
        nino3_anomaly_degC=w.NINO3.to_numpy(), nino34_anomaly_degC=w.NINO34.to_numpy(),
        nino4_anomaly_degC=w.NINO4.to_numpy(), standardized_nino3=z.NINO3.to_numpy(),
        standardized_nino4=z.NINO4.to_numpy(), standardized_nino4_minus_nino3=contrast.to_numpy(),
        selected_el_nino_winter=(w.NINO34 >= .5).to_numpy(),
        winter_type=np.where(w.NINO34 >= .5, np.where(contrast > 0, 'CP', 'EP'), '')))
    selected = catalogue[catalogue.selected_el_nino_winter].copy()
    assert len(selected) == 12 and (contrast != 0).all()
    for key in ['NINO3', 'NINO4']:
        mean = sum(w[key])/45
        sd = (sum((x-mean)**2 for x in w[key])/44)**.5
        np.testing.assert_allclose(z[key], [(x-mean)/sd for x in w[key]], atol=1e-14)
    parameters, classifications, scenarios = [], [], []
    for label, start, end in [('1982-2026', 1982, 2026), ('1991-2020', 1991, 2020),
                              ('1982-2004', 1982, 2004)]:
        reference = w.loc[start:end, ['NINO3', 'NINO4']]
        standardized = (w[reference.columns]-reference.mean())/reference.std(ddof=1)
        difference = standardized.NINO4-standardized.NINO3
        frame = selected.copy()
        frame['winter_type'] = np.where(difference.loc[frame.winter_year].to_numpy() > 0, 'CP', 'EP')
        changed = frame.loc[frame.winter_type != selected.winter_type, 'winter_year'].tolist()
        table = count_table(frame)
        p, _ = exact_test(table)
        scenarios.append(dict(scenario=label, earlier_central=table[1, 0], earlier_eastern=table[1, 1],
            later_central=table[0, 0], later_eastern=table[0, 1], selected_winters=int(table.sum()),
            changed_or_omitted_winters=';'.join(map(str, changed)), two_sided_p=p))
        for key in reference:
            parameters.append(dict(reference_period=label, index=key, winters=len(reference),
                mean_degC=reference[key].mean(), sample_standard_deviation_degC=reference[key].std(ddof=1)))
        for row in frame.itertuples():
            classifications.append(dict(reference_period=label, winter_year=row.winter_year,
                winter_type=row.winter_type, standardized_nino4_minus_nino3=difference.loc[row.winter_year]))
    table = count_table(selected)
    assert table.tolist() == [[5, 1], [2, 4]]
    p, enumeration = exact_test(table)
    assert abs(p-8/33) < 1e-15
    conditional = odds_ratio(table, kind='conditional')
    interval = conditional.confidence_interval(.95)
    xs = enumeration.later_central.to_numpy()
    weights = np.array([comb(7, int(x))*comb(5, 6-int(x)) for x in xs])
    def tilted(log_odds):
        logw = np.log(weights)+xs*log_odds
        probabilities = np.exp(logw-logw.max())
        return probabilities/probabilities.sum()
    estimate = np.exp(brentq(lambda v: np.dot(xs, tilted(v))-5, -30, 30))
    low = np.exp(brentq(lambda v: tilted(v)[xs >= 5].sum()-.025, -30, 30))
    high = np.exp(brentq(lambda v: tilted(v)[xs <= 5].sum()-.025, -30, 30))
    np.testing.assert_allclose([estimate, low, high], [conditional.statistic, interval.low, interval.high], rtol=1e-10)
    inference = pd.DataFrame([dict(two_sided_p=p, sample_odds_ratio=float(fisher_exact(table).statistic),
        conditional_odds_ratio=conditional.statistic, exact_95_percent_lower=interval.low,
        exact_95_percent_upper=interval.high)])
    for year in [2015, 2016]:
        table = count_table(selected[selected.winter_year != year])
        value, _ = exact_test(table)
        scenarios.append(dict(scenario=f'omit_{year}', earlier_central=table[1, 0], earlier_eastern=table[1, 1],
            later_central=table[0, 0], later_eastern=table[0, 1], selected_winters=int(table.sum()),
            changed_or_omitted_winters=str(year), two_sided_p=value))
    weekly = pd.read_excel(source, 'Weekly_SST', skiprows=2).set_index('Week_ending').NINO34
    assert weekly.notna().all() and weekly.index.is_unique
    monthly = weekly.groupby(pd.DatetimeIndex(weekly.index).to_period('M')).mean()
    months = monthly.loc['1981-12':'2026-02']
    assert months.index.equals(pd.period_range('1981-12', '2026-02', freq='M'))
    rolling = months.rolling(3, center=True, min_periods=3).mean().dropna()
    flags = rolling >= .5
    ids = (flags & ~flags.shift(fill_value=False)).cumsum().where(flags, 0)
    windows = pd.DataFrame(dict(centre_month=rolling.index.astype(str),
        nino34_three_month_anomaly_degC=rolling.to_numpy(), warm_run_id=ids.to_numpy()))
    for year in w.index:
        np.testing.assert_allclose(rolling.loc[f'{year}-01'], w.loc[year, 'NINO34'], atol=1e-14)
    runs = []
    for rid, group in windows[windows.warm_run_id > 0].groupby('warm_run_id'):
        years = [int(y) for y in selected.winter_year if f'{y}-01' in set(group.centre_month)]
        runs.append(dict(warm_run_id=rid, first_centre_month=group.centre_month.iloc[0],
            last_centre_month=group.centre_month.iloc[-1], consecutive_three_month_windows=len(group),
            selected_winter_count=len(years), selected_winter_years=';'.join(map(str, years)),
            touches_analysis_boundary=group.index[0] == 0 or group.index[-1] == len(windows)-1))
    runs = pd.DataFrame(runs)
    assert runs.loc[runs.selected_winter_count > 1, 'selected_winter_years'].tolist() == ['2015;2016']
    return {
        'el_nino_winter_classification.csv': catalogue,
        'el_nino_standardization_parameters.csv': pd.DataFrame(parameters),
        'el_nino_standardization_sensitivity.csv': pd.DataFrame(classifications),
        'el_nino_count_sensitivity.csv': pd.DataFrame(scenarios),
        'el_nino_fisher_inference.csv': inference,
        'el_nino_fisher_table_probabilities.csv': enumeration,
        'el_nino_three_month_windows.csv': windows,
        'el_nino_continuous_warm_runs.csv': runs}, selected


def plot(selected):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42,
        'savefig.dpi': 300, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 4.8), gridspec_kw={'width_ratios': [1.3, 1]})
    fig.subplots_adjust(left=.10, right=.98, top=.75, bottom=.18, wspace=.46)
    fig.suptitle('El Niño winter classification', y=.975, fontsize=13)
    fig.text(.5, .923, '12 selected winters from 45 complete winters, 1982–2026', ha='center', fontsize=9)
    colors = {'CP': '#2166AC', 'EP': '#B2182B'}
    labels = {'CP': 'Central-Pacific type', 'EP': 'Eastern-Pacific type'}
    for kind, marker in [('CP', 'o'), ('EP', 's')]:
        mask = selected.winter_type == kind
        x = selected.loc[mask, 'standardized_nino4_minus_nino3'].to_numpy()
        y = np.arange(len(selected))[mask]
        points = axes[0].scatter(x, y, s=24, marker=marker, color=colors[kind], label=labels[kind], zorder=3)
        np.testing.assert_allclose(points.get_offsets(), np.column_stack([x, y]))
    for y, row in enumerate(selected.itertuples()):
        x = row.standardized_nino4_minus_nino3
        axes[0].text(x+(.06 if x >= 0 else -.06), y, f'{x:+.3f}',
                     ha='left' if x >= 0 else 'right', va='center', fontsize=7)
    axes[0].axvline(0, color='.45', ls='--', lw=.8)
    axes[0].axhline(5.5, color='.5', ls=':', lw=.8)
    axes[0].set(yticks=range(12), yticklabels=selected.winter_year, ylim=(11.6, -.6),
        xlim=(-2.5, 1.1), ylabel='Winter (January year)',
        xlabel='Standardized Niño 4 minus Niño 3\n(dimensionless)')
    axes[0].set_title('(a) Selected winters', fontsize=10, loc='left')
    table = count_table(selected)
    for i, kind in enumerate(['CP', 'EP']):
        bars = axes[1].bar(np.arange(2)+(i-.5)*.32, table[::-1, i], width=.32,
                           color=colors[kind], alpha=.85)
        axes[1].bar_label(bars, padding=3)
        np.testing.assert_array_equal([b.get_height() for b in bars], table[::-1, i])
    axes[1].set(xticks=[0, 1], xticklabels=['1982–2004', '2005–2026'],
        ylim=(0, 6.2), yticks=range(7), ylabel='Number of selected winters')
    axes[1].set_title('(b) Counts by period', fontsize=10, loc='left')
    for ax in axes:
        ax.tick_params(labelsize=8)
        ax.grid(axis='y' if ax is axes[1] else 'x', alpha=.15, lw=.5)
        ax.set_axisbelow(True)
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=2, frameon=False,
               loc='upper center', bbox_to_anchor=(.5, .875), fontsize=9)
    for extension in ['png', 'pdf']:
        fig.savefig(FIGURES /f'figure_08_el_nino_winter_classification.{extension}', facecolor='white')
    plt.close(fig)
