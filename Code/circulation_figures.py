"""Reproduce atmospheric-circulation Figures 2, S5 and S6 from verified inputs.

Run with Python, numpy, pandas, scipy, statsmodels, matplotlib and openpyxl.
Existing data files are checked, not overwritten. Numerical methods and source
aggregation are shared with the scripts for supplementary Tables S3-S9 and S2.
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
NAMES = {'AO': 'Arctic Oscillation', 'NAO': 'North Atlantic Oscillation'}
BLUE, RED, GREEN, INK = '#2166AC', '#B2182B', '#287A58', '#262626'


def verify():
    aggregation = runpy.run_path(str(ROOT/'Code'/'winter_aggregation.py'))
    methods = runpy.run_path(str(ROOT/'Code'/'trend_analysis.py'))
    winters = aggregation['calculate']()[0]
    recalculated = methods['calculate'](winters)
    for filename, frame in recalculated.items():
        saved = pd.read_csv(ROOT / 'Data' / filename)
        assert list(saved.columns) == list(frame.columns)
        pd.testing.assert_frame_equal(saved, frame, check_dtype=False, atol=1e-12, rtol=1e-10)
    for key in NAMES:
        series = winters.loc[1951:2026, key]
        assert series.notna().all() and len(series) == 76
        # Direct pair comparisons independently check Kendall's tau-a.
        pairs = [(j-i, b-a) for i, a in series.items() for j, b in series.items() if j > i]
        row = recalculated['winter_trend_results.csv'].query('index == @key and scope == "own_record"').iloc[0]
        np.testing.assert_allclose(np.mean([np.sign(dy) for dx, dy in pairs]), row.tau_a, atol=1e-14)
        np.testing.assert_allclose(np.median([dy/dx for dx, dy in pairs])*10, row.sen_per_decade, atol=1e-14)
        counts = np.unique(series.to_numpy(), return_counts=True)[1]
        tied_pairs = np.sum(counts*(counts-1)/2)
        tau_b_to_a = np.sqrt((len(pairs)-tied_pairs)/len(pairs))
        np.testing.assert_allclose(stats.kendalltau(series.index, series).statistic*tau_b_to_a,
                                   row.tau_a, atol=1e-14)
    return winters, recalculated


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#AAAAAA')
    ax.tick_params(labelsize=8, length=3)
    ax.set_axisbelow(True)


def save(fig, stem):
    for extension in ['png', 'pdf']:
        fig.savefig(FIGURES / f'{stem}.{extension}', facecolor='white')
    plt.close(fig)


def records(winters, outputs):
    trends = outputs['winter_trend_results.csv'].query('scope == "own_record"').set_index('index')
    periods = outputs['winter_period_mean_comparisons.csv'].set_index('index')
    fig, axes = plt.subplots(2, 1, figsize=(7, 5.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=.11, right=.97, top=.82, bottom=.11, hspace=.48)
    fig.suptitle('Winter atmospheric circulation, 1951–2026', y=.98, fontsize=12)
    fig.legend(handles=[Line2D([], [], color=INK, lw=1, label='Winter mean'),
                        Line2D([], [], color=INK, ls='--', lw=1.4, label='Theil–Sen trend'),
                        Line2D([], [], color=GREEN, lw=2, label='Mean within each period')],
               loc='upper center', bbox_to_anchor=(.52, .94), ncol=3, frameon=False, fontsize=8)
    for i, (ax, key) in enumerate(zip(axes, NAMES)):
        s = winters.loc[1951:2026, key]
        years, values = s.index.to_numpy(), s.to_numpy()
        row, period = trends.loc[key], periods.loc[key]
        ax.fill_between(years, 0, values, where=values >= 0, interpolate=True, color=RED, alpha=.30)
        ax.fill_between(years, 0, values, where=values <= 0, interpolate=True, color=BLUE, alpha=.30)
        line, = ax.plot(years, values, color=INK, lw=.8)
        np.testing.assert_array_equal(line.get_ydata(), values)
        slope = row.sen_per_decade / 10
        ax.plot(years, slope*years + np.median(values-slope*years), '--', color=INK, lw=1.3)
        ax.hlines([period.early_mean, period.late_mean], [1951, 1989], [1988, 2026], color=GREEN, lw=2)
        ax.axvline(1988.5, color='#888888', lw=.7, ls=':')
        ax.axhline(0, color='#777777', lw=.5)
        ax.set_title(f'({chr(97+i)}) {NAMES[key]}\n'
                     f'Sen slope: {row.sen_per_decade:+.3f} units/decade; adjusted p = {row.q_bh:.3g}',
                     loc='left', fontsize=9, pad=7)
        ax.set(ylabel='Index (dimensionless)', ylim=(-3.7, 2.7), xlim=(1951, 2026),
               xticks=[1951, 1965, 1980, 1989, 2000, 2015, 2026])
        style(ax)
    axes[-1].set_xlabel('Winter year (January year)')
    save(fig, 'figure_02_winter_atmospheric_circulation')


def period_comparison(winters, outputs):
    fig = plt.figure(figsize=(7, 6.3))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.6, 1], left=.15, right=.96,
                           top=.87, bottom=.105, hspace=.61, wspace=.32)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    fig.suptitle('Winter circulation in two periods', y=.98, fontsize=12)
    fig.legend(handles=[Line2D([], [], marker='o', ls='', ms=4, color=INK, label='Individual winter'),
                        Line2D([], [], marker='D', ls='', ms=5, color=GREEN, label='Period mean')],
               loc='upper center', bbox_to_anchor=(.51, .948), ncol=2, frameon=False, fontsize=8)
    rng = np.random.default_rng(19511989)
    for i, (ax, key) in enumerate(zip(axes, NAMES)):
        groups = [winters.loc[1951:1988, key].to_numpy(), winters.loc[1989:2026, key].to_numpy()]
        boxes = ax.boxplot(groups, positions=[0, 1], widths=.42, whis=1.5, patch_artist=True,
                           showfliers=False, medianprops={'color': INK, 'linewidth': 1.4})
        for j, (values, color, box) in enumerate(zip(groups, [BLUE, RED], boxes['boxes'])):
            box.set(facecolor=color, alpha=.16)
            points = ax.scatter(j+rng.uniform(-.12, .12, len(values)), values, s=13,
                                color=color, alpha=.70, linewidths=0, zorder=3)
            np.testing.assert_array_equal(points.get_offsets()[:, 1], values)
            ax.scatter(j, values.mean(), marker='D', s=33, color=GREEN, edgecolor='white', linewidth=.5, zorder=4)
        ax.axhline(0, color='#888888', lw=.5)
        ax.set(xticks=[0, 1], xticklabels=['1951–1988\n38 winters', '1989–2026\n38 winters'], ylim=(-3.7, 3.0))
        ax.set_title(f'({chr(97+i)}) {NAMES[key]}', loc='left', fontsize=9)
        style(ax)
    axes[0].set_ylabel('Winter index (dimensionless)')
    ax = fig.add_subplot(grid[1, :])
    rows = outputs['winter_period_difference_uncertainty.csv'].query('max_lag_years == 3').set_index('index')
    for y, key in enumerate(NAMES):
        row = rows.loc[key]
        ax.errorbar(row.difference, y, xerr=[[row.difference-row.ci95_low], [row.ci95_high-row.difference]],
                    fmt='D', color=GREEN, capsize=4, ms=5)
        ax.text(row.difference, y-.21, f'{row.difference:+.3f} [{row.ci95_low:+.3f}, {row.ci95_high:+.3f}]',
                ha='center', fontsize=8)
    ax.axvline(0, color=INK, ls='--', lw=.7)
    ax.set(yticks=[0, 1], yticklabels=['Arctic', 'North Atlantic'], ylim=(1.6, -.6), xlim=(-.05, 1.4),
           xlabel='Later minus earlier mean (dimensionless index units)')
    ax.set_title('(c) Mean differences with 95% confidence intervals', loc='left', fontsize=9, pad=8)
    style(ax)
    save(fig, 'figure_s05_winter_circulation_period_comparison')


def sensitivity(outputs):
    fig, axes = plt.subplots(1, 2, figsize=(7, 4.5), sharex=True, sharey=True)
    fig.subplots_adjust(left=.17, right=.97, top=.79, bottom=.17, wspace=.18)
    fig.suptitle('Sensitivity of winter trends to the study period', y=.98, fontsize=12)
    fig.text(.55, .912, 'Theil–Sen slopes and conventional 95% confidence intervals', ha='center', fontsize=8)
    for i, (ax, key) in enumerate(zip(axes, NAMES)):
        rows = outputs['winter_trend_period_sensitivity.csv'].query('index == @key')
        for y, row in enumerate(rows.itertuples()):
            estimate = row.sen_per_decade
            ax.errorbar(estimate, y, xerr=[[estimate-row.scipy_ci95_low], [row.scipy_ci95_high-estimate]],
                        fmt='o', color=INK, capsize=4, ms=4)
            probability = f'{row.p_tfpw:.7f}' if row.p_tfpw < .0001 else f'{row.p_tfpw:.3g}'
            ax.text(.97, y-.18, f'p = {probability}', transform=ax.get_yaxis_transform(),
                    ha='right', fontsize=8)
        ax.axvline(0, color='#888888', ls='--', lw=.8)
        ax.set_title(f'({chr(97+i)}) {NAMES[key]}', loc='left', fontsize=9, pad=10)
        ax.set(xlim=(-.4, .52), ylim=(3.5, -.65), yticks=range(4), yticklabels=rows.period,
               xlabel='Slope (index units/decade)')
        style(ax)
    save(fig, 'figure_s06_winter_trend_period_sensitivity')
