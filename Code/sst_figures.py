"""Reproduce Figures 3, 4 and S7 without modifying observations or result files.

Requires numpy, pandas, scipy, statsmodels, matplotlib and openpyxl.
Source aggregation and statistical methods are shared with the existing scripts
for Figure S2 and Tables S3-S9. All paths are relative to this file.
"""
from pathlib import Path
from hashlib import sha256
import runpy

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT/'Figures'
KEYS = ['TNA', 'AMO', 'NINO12', 'NINO3', 'NINO34', 'NINO4']
LABELS = ['Tropical North\nAtlantic', 'Atlantic Multidecadal\nOscillation (detrended)',
          'Niño 1+2', 'Niño 3', 'Niño 3.4', 'Niño 4']
BLUE, RED, INK = '#2166AC', '#B2182B', '#262626'


def calculate():
    aggregation = runpy.run_path(str(ROOT/'Code'/'winter_aggregation.py'))
    methods = runpy.run_path(str(ROOT/'Code'/'trend_analysis.py'))
    winters = aggregation['calculate']()[0]
    outputs = methods['calculate'](winters)
    for filename, frame in outputs.items():
        pd.testing.assert_frame_equal(pd.read_csv(ROOT / 'Data' / filename), frame,
                                      check_dtype=False, atol=1e-12, rtol=1e-10)
    for row in outputs['winter_trend_results.csv'].query('index in @KEYS').itertuples():
        s = winters.loc[row.start:row.end, row.index].dropna()
        slopes = [(b-a)/(j-i) for i, a in s.items() for j, b in s.items() if j > i]
        np.testing.assert_allclose(np.median(slopes)*10, row.sen_per_decade, atol=1e-14)
    # Scalar regression formulas independently verify paired trend uncertainty.
    x = (np.arange(1982, 2027)-1982)/10
    centered = x-x.mean()
    for row in outputs['atlantic_pacific_trend_contrasts.csv'].itertuples():
        y = (winters.loc[1982:2026, 'TNA']-winters.loc[1982:2026, row.pacific_index]).to_numpy()
        slope = np.dot(centered, y)/np.dot(centered, centered)
        residual = y-y.mean()-slope*centered
        scores = residual*centered
        meat = np.dot(scores, scores)
        for lag in range(1, row.hac_lag_years+1):
            meat += 2*(1-lag/(row.hac_lag_years+1))*np.dot(scores[lag:], scores[:-lag])
        se = np.sqrt(meat*45/43/np.dot(centered, centered)**2)
        half = stats.t.ppf(.975, 43)*se
        np.testing.assert_allclose([slope, slope-half, slope+half, 2*stats.t.sf(abs(slope/se), 43)],
            [row.slope_difference_per_decade, row.ci95_low, row.ci95_high, row.p], atol=1e-12)
    # Reconstruct the circular resamples separately, including end wrapping.
    for row in outputs['nino34_winter_variability_uncertainty.csv'].itertuples():
        rng = np.random.default_rng(row.seed)
        standard_deviations = []
        for a, b in [(1982, 2004), (2005, 2026)]:
            values = winters.loc[a:b, 'NINO34'].to_numpy()
            starts = rng.integers(0, len(values), (row.draws, int(np.ceil(len(values)/row.block_length_years))))
            samples = np.empty((row.draws, len(values)))
            for column in range(len(values)):
                block, offset = divmod(column, row.block_length_years)
                samples[:, column] = values[(starts[:, block]+offset) % len(values)]
            mean = samples.sum(axis=1)/len(values)
            standard_deviations.append(np.sqrt(((samples-mean[:, None])**2).sum(axis=1)/(len(values)-1)))
        ratio = standard_deviations[1]/standard_deviations[0]
        np.testing.assert_allclose(np.quantile(ratio, [.025, .975], method='linear'),
                                   [row.ci95_low, row.ci95_high], atol=1e-13)
    return winters, outputs


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#AAAAAA')
    ax.tick_params(labelsize=8, length=3)


def interval(ax, estimate, low, high, y, marker='o'):
    line, caps, bars = ax.errorbar(estimate, y, xerr=[[estimate-low], [high-estimate]],
                                  fmt=marker, color=INK, capsize=4, ms=4.5)
    np.testing.assert_allclose(np.asarray(line.get_xdata(), dtype=float), [estimate])
    np.testing.assert_allclose(bars[0].get_segments()[0], [[low, y], [high, y]])


def save(fig, stem):
    for ext in ['png', 'pdf']:
        fig.savefig(FIGURES / f'{stem}.{ext}', facecolor='white')
    plt.close(fig)


def trend_comparison(outputs):
    fig, axes = plt.subplots(1, 2, figsize=(7, 5.4), sharex=True, sharey=True)
    fig.subplots_adjust(left=.29, right=.975, top=.77, bottom=.13, wspace=.17)
    fig.suptitle('Winter sea-surface temperature trends', y=.975, fontsize=12)
    for ax, scope, title in zip(axes, ['own_record', 'common_1983_2022'],
                                ['(a) Individual records', '(b) Common period\n1983–2022']):
        rows = outputs['winter_trend_results.csv'].query('scope == @scope').set_index('index')
        for y, key in enumerate(KEYS):
            row = rows.loc[key]
            interval(ax, row.sen_per_decade, row.scipy_ci95_low, row.scipy_ci95_high, y,
                     'D' if key == 'AMO' else 'o')
        ax.axvline(0, color='#888888', ls='--', lw=.8)
        ax.set(yticks=range(6), yticklabels=LABELS, ylim=(5.6, -.6), xlim=(-.43, .39),
               xticks=[-.4, -.2, 0, .2], xlabel='Sen slope (°C/decade)')
        ax.set_title(title, loc='left', fontsize=9, pad=10)
        style(ax)
    fig.text(.64, .89, 'Points and bars: estimates and conventional 95% intervals', ha='center', fontsize=8)
    save(fig, 'figure_03_sst_trends_by_study_period')


def contrasts_and_variability(outputs):
    fig = plt.figure(figsize=(7, 6.4))
    grid = fig.add_gridspec(2, 2, left=.15, right=.97, top=.86, bottom=.115,
                           height_ratios=[1.15, 1], hspace=.63, wspace=.45)
    fig.suptitle('Atlantic–Pacific trend contrasts and Pacific variability', y=.98, fontsize=11)
    ax = fig.add_subplot(grid[0, :])
    rows = outputs['atlantic_pacific_trend_contrasts.csv'].query('hac_lag_years == 3')
    for y, row in enumerate(rows.itertuples()):
        interval(ax, row.slope_difference_per_decade, row.ci95_low, row.ci95_high, y)
        ax.text(.99, y-.19, f'adjusted p = {row.q_bh:.4f}', transform=ax.get_yaxis_transform(),
                ha='right', fontsize=8)
    ax.axvline(0, color='#888888', ls='--', lw=.8)
    ax.set(yticks=range(3), yticklabels=['Niño 3', 'Niño 3.4', 'Niño 4'], ylim=(2.6, -.6),
           xlim=(-.03, .52), xlabel='Difference in fitted trend (°C/decade)', ylabel='Pacific region')
    ax.set_title('(a) Tropical North Atlantic minus Pacific, 1982–2026', loc='left', fontsize=9)
    style(ax)
    row = outputs['nino34_winter_variability_uncertainty.csv'].query('block_length_years == 3').iloc[0]
    ax = fig.add_subplot(grid[1, 0])
    heights = [row.early_sd, row.late_sd]
    bars = ax.bar([0, 1], heights, width=.55, color=[BLUE, RED], alpha=.65)
    np.testing.assert_allclose([b.get_height() for b in bars], heights)
    for b, n in zip(bars, [int(row.n_early), int(row.n_late)]):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+.04, f'{b.get_height():.3f}\n{n} winters',
                ha='center', fontsize=8)
    ax.set(xticks=[0, 1], xticklabels=['1982–2004', '2005–2026'], ylim=(0, 1.6),
           ylabel='Sample standard deviation (°C)')
    ax.set_title('(b) Niño 3.4 winter variability', loc='left', fontsize=9)
    style(ax)
    ax = fig.add_subplot(grid[1, 1])
    interval(ax, row.sd_ratio, row.ci95_low, row.ci95_high, 0)
    ax.axvline(1, color='#888888', ls='--', lw=.8)
    ax.text(.5, .78, f'{row.sd_ratio:.3f} [{row.ci95_low:.3f}, {row.ci95_high:.3f}]',
            ha='center', transform=ax.transAxes, fontsize=8, bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1})
    ax.set(xlim=(.5, 1.45), ylim=(-.6, .6), yticks=[], xlabel='Later / earlier standard deviation')
    ax.set_title('(c) Variability ratio\n95% bootstrap interval', loc='left', fontsize=9)
    style(ax)
    save(fig, 'figure_04_atlantic_pacific_contrasts_and_variability')


def winter_records(winters, outputs):
    fig, axes = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True, sharey=True)
    fig.subplots_adjust(left=.115, right=.97, top=.82, bottom=.115, hspace=.35)
    fig.suptitle('Winter sea-surface temperature anomalies, 1982–2026', y=.98, fontsize=11.5)
    fig.legend(handles=[Line2D([], [], color=INK, lw=1, label='Winter mean'),
                        Line2D([], [], color=INK, ls='--', lw=1.3, label='Theil–Sen trend')],
               loc='upper center', bbox_to_anchor=(.52, .94), ncol=2, frameon=False, fontsize=8)
    rows = outputs['winter_trend_results.csv'].query('scope == "own_record"').set_index('index')
    for ax, key, name in zip(axes, ['TNA', 'NINO34'], ['(a) Tropical North Atlantic', '(b) Niño 3.4']):
        s = winters.loc[1982:2026, key]
        assert len(s) == 45 and s.notna().all() and s.min() > -2.7 and s.max() < 3
        years, values = s.index.to_numpy(), s.to_numpy()
        ax.fill_between(years, 0, values, where=values>=0, interpolate=True, color=RED, alpha=.3)
        ax.fill_between(years, 0, values, where=values<=0, interpolate=True, color=BLUE, alpha=.3)
        line, = ax.plot(years, values, lw=.85, color=INK)
        np.testing.assert_array_equal(line.get_ydata(), values)
        slope = rows.loc[key].sen_per_decade/10
        ax.plot(years, slope*years+np.median(values-slope*years), '--', lw=1.3, color=INK)
        ax.axhline(0, color='#888888', lw=.5)
        ax.set(ylabel='Temperature anomaly (°C)', ylim=(-2.7, 3), xlim=(1982, 2026),
               xticks=[1982, 1990, 2000, 2010, 2020, 2026])
        ax.set_title(name, loc='left', fontsize=9)
        style(ax)
    axes[-1].set_xlabel('Winter year (January year)')
    save(fig, 'figure_s07_tna_nino34_winter_anomalies')
