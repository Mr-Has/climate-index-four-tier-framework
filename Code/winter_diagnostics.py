"""Reproduce the winter distribution and autocorrelation figures from the workbook."""
from pathlib import Path
from hashlib import sha256
import runpy
import sys
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cbook import boxplot_stats
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from statsmodels.tsa.stattools import acf

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT/'Figures'
KEYS = ['AO', 'NAO', 'PNA', 'PDO', 'SOI', 'AMO', 'NINO34', 'TNA']
NAMES = ['Arctic Oscillation', 'North Atlantic Oscillation',
         'Pacific–North American pattern', 'Pacific Decadal Oscillation',
         'Southern Oscillation Index', 'Atlantic Multidecadal Oscillation\n(detrended source)',
         'Niño 3.4 temperature anomaly', 'Tropical North Atlantic\ntemperature anomaly']


def calculate():
    native = runpy.run_path(str(ROOT/'Code/winter_aggregation.py'))['calculate']()[1]
    saved = pd.read_csv(ROOT/'Data/winter_climate_indices_1983_2022.csv', index_col='winter_year')
    np.testing.assert_allclose(native, saved, atol=1e-13, rtol=0)
    stats = pd.read_csv(ROOT/'Data/winter_climate_index_statistics_1983_2022.csv').set_index('index')
    saved_acf = pd.read_csv(ROOT/'Data/winter_index_autocorrelations.csv').set_index(['index', 'lag_winters'])
    boxes, coefficients = [], {}
    for key in KEYS:
        x = native[key].to_numpy()
        ordered = sorted(x)
        qs = []
        for probability in [.25, .5, .75]:
            position = 39*probability
            lo, hi = int(np.floor(position)), int(np.ceil(position))
            qs.append(ordered[lo]+(position-lo)*(ordered[hi]-ordered[lo]))
        np.testing.assert_allclose(qs, np.quantile(x, [.25, .5, .75], method='linear'), atol=1e-14)
        q1, median, q3 = qs
        spread = q3-q1
        inside = (x >= q1-1.5*spread) & (x <= q3+1.5*spread)
        box = boxplot_stats(x, whis=1.5)[0]
        np.testing.assert_allclose([box[k] for k in ['q1', 'med', 'q3', 'whislo', 'whishi']],
                                  [q1, median, q3, x[inside].min(), x[inside].max()], atol=1e-14)
        np.testing.assert_array_equal(np.sort(box['fliers']), np.sort(x[~inside]))
        np.testing.assert_allclose([x.mean(), x.std(ddof=1), x.min(), q1, median, q3, x.max()],
            stats.loc[key, ['mean', 'sample_standard_deviation', 'minimum', 'first_quartile',
                            'median', 'third_quartile', 'maximum']].astype(float), atol=1e-13)
        boxes.append(box)
        centered = x-x.mean()
        values = np.array([centered[k:]@centered[:40-k]/(centered@centered) for k in range(11)])
        np.testing.assert_allclose(values, acf(x, nlags=10, adjusted=False, fft=False), atol=1e-14)
        np.testing.assert_allclose(values, np.correlate(centered, centered, 'full')[39:50]/(centered@centered), atol=1e-14)
        np.testing.assert_allclose(values, acf(centered/x.std(ddof=1), nlags=10, adjusted=False, fft=False), atol=1e-14)
        np.testing.assert_allclose(values[1:], saved_acf.loc[key, 'autocorrelation'], atol=1e-13)
        np.testing.assert_array_equal(saved_acf.loc[key, 'contributing_pairs'], 40-np.arange(1, 11))
        coefficients[key] = values[1:]
        print(key, 'beyond-whisker winters:', ', '.join(map(str, native.index[~inside])) or 'none',
              'lag-one autocorrelation:', f'{values[1]:.6f}')
    return native, boxes, coefficients


def save(fig, name):
    for extension in ['png', 'pdf']:
        fig.savefig(FIGURES /f'{name}.{extension}', facecolor='white')
    plt.close(fig)


def distributions(native, boxes):
    fig, axes = plt.subplots(4, 2, figsize=(7.5, 7.2))
    fig.subplots_adjust(left=.07, right=.985, top=.80, bottom=.08, hspace=1.28, wspace=.27)
    fig.suptitle('Distributions of winter climate indices', y=.985, fontsize=13)
    fig.text(.5, .95, 'December–February, 1983–2022; 40 winters per index', ha='center', fontsize=9)
    fig.text(.5, .923, 'Original units; horizontal scales differ between panels', ha='center', fontsize=8.5)
    fig.legend(handles=[Patch(facecolor='#dbe8f1', edgecolor='#2166AC', label='Middle 50%'),
        Line2D([0], [0], color='#262626', label='Median'),
        Line2D([0], [0], color='none', marker='D', markerfacecolor='#B2182B', markeredgecolor='white', label='Mean'),
        Line2D([0], [0], color='none', marker='o', markerfacecolor='#2166AC', label='One winter')],
        loc='upper center', bbox_to_anchor=(.5, .904), ncol=4, frameon=False, fontsize=8)
    rng = np.random.default_rng(19832022)
    for i, (ax, key, name, box) in enumerate(zip(axes.flat, KEYS, NAMES, boxes)):
        x = native[key].to_numpy()
        artists = ax.bxp([box], positions=[.65], widths=.24, vert=False, showfliers=False,
            showmeans=True, patch_artist=True, manage_ticks=False,
            boxprops=dict(facecolor='#dbe8f1', edgecolor='#2166AC'),
            medianprops=dict(color='#262626', linewidth=1.3),
            meanprops=dict(marker='D', markerfacecolor='#B2182B', markeredgecolor='white', markersize=4))
        y = .23+rng.uniform(-.045, .045, len(x))
        points = ax.scatter(x, y, s=10, color='#2166AC', alpha=.75, edgecolor='white', linewidth=.2)
        np.testing.assert_array_equal(points.get_offsets()[:, 0], x)
        np.testing.assert_allclose(artists['medians'][0].get_xdata(), box['med'], atol=1e-14)
        np.testing.assert_allclose(artists['means'][0].get_xdata(), x.mean(), atol=1e-14)
        for year, value, point_y in zip(native.index, x, y):
            if value < box['whislo'] or value > box['whishi']:
                ax.annotate(str(year), (value, point_y), xytext=(0, -10), textcoords='offset points',
                            ha='center', fontsize=6.8)
        ax.set_title(f'({chr(97+i)}) {name}', loc='left', fontsize=8.5, pad=13)
        ax.text(.99, 1.0, f'Mean {x.mean():+.3f}; median {box["med"]:+.3f}',
                transform=ax.transAxes, ha='right', fontsize=7, va='bottom')
        unit = 'Temperature anomaly (°C)' if key in ['AMO', 'NINO34', 'TNA'] else (
            'Provider index scale' if key == 'PDO' else 'Index (dimensionless)')
        width = x.max()-x.min()
        ax.set(xlim=(x.min()-.12*width, x.max()+.12*width), ylim=(-.15, 1.05), yticks=[])
        ax.set_xlabel(unit, fontsize=8, labelpad=4)
        ax.tick_params(axis='x', labelsize=7.5, length=2)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.grid(axis='x', color='#dddddd', lw=.5)
        ax.set_axisbelow(True)
    save(fig, 'figure_s03_winter_climate_index_distributions')


def autocorrelations(coefficients):
    fig, axes = plt.subplots(4, 2, figsize=(7.5, 7.2), sharex=True, sharey=True)
    fig.subplots_adjust(left=.095, right=.975, top=.75, bottom=.08, hspace=.70, wspace=.24)
    fig.suptitle('Autocorrelation of winter climate indices', y=.985, fontsize=13)
    fig.text(.5, .95, '40 December–February means, labelled 1983–2022', ha='center', fontsize=9)
    fig.text(.5, .916, 'Winter 1983: December 1982–February 1983; winter 2022: December 2021–February 2022',
             ha='center', fontsize=7.8)
    fig.text(.5, .882, 'Lag 1: 1983 with 1984, …, 2021 with 2022 (39 pairs)', ha='center', fontsize=8)
    fig.text(.5, .854, 'Lag 10: 1983 with 1993, …, 2012 with 2022 (30 pairs)', ha='center', fontsize=8)
    fig.text(.5, .822, 'Each coefficient uses all pairs at that lag; winters are not divided into groups.',
             ha='center', fontsize=8)
    lags = np.arange(1, 11)
    for i, (ax, key, name) in enumerate(zip(axes.flat, KEYS, NAMES)):
        values = coefficients[key]
        colors = np.where(values >= 0, '#B2182B', '#2166AC')
        stems = ax.vlines(lags, 0, values, color=colors, lw=1.2)
        points = ax.scatter(lags, values, color=colors, s=14, zorder=3)
        np.testing.assert_array_equal(points.get_offsets(), np.column_stack([lags, values]))
        for k, v, segment in zip(lags, values, stems.get_segments()):
            np.testing.assert_array_equal(segment, [[k, 0], [k, v]])
        ax.axhline(0, color='.35', lw=.65)
        ax.set_title(f'({chr(97+i)}) {name}', loc='left', fontsize=8.5, pad=6)
        ax.text(.98, .91, f'Lag 1: {values[0]:+.3f}', transform=ax.transAxes, ha='right', fontsize=7)
        ax.set(ylim=(-1, 1), xlim=(.5, 10.5), xticks=lags, yticks=[-1, 0, 1])
        ax.tick_params(labelsize=7.5, length=2, labelbottom=True, labelleft=True)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', color='#dddddd', lw=.5)
    fig.text(.02, .42, 'Sample autocorrelation (dimensionless)', rotation=90, va='center', fontsize=9)
    fig.text(.53, .02, 'Years between paired winters (lag)', ha='center', fontsize=9)
    save(fig, 'figure_s04_winter_climate_index_autocorrelation')
