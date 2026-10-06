"""Reproduce winter aggregation, standardization and the winter-series figure."""
from pathlib import Path
from hashlib import sha256

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT/'Figures'
SOURCE = ROOT / 'Data' / 'Climate_Indices_Data.xlsx'
KEYS = ['AO', 'NAO', 'PNA', 'PDO', 'SOI', 'AMO', 'NINO34', 'TNA']
NAMES = ['Arctic Oscillation', 'North Atlantic Oscillation',
         'Pacific–North American pattern', 'Pacific Decadal Oscillation',
         'Southern Oscillation Index', 'Atlantic Multidecadal Oscillation',
         'Niño 3.4 temperature anomaly', 'Tropical North Atlantic temperature anomaly']
UNITS = ['dimensionless'] * 3 + ['provider scale', 'dimensionless', '°C', '°C', '°C']


def calculate():
    monthly = pd.read_excel(SOURCE, 'Monthly', skiprows=2)
    monthly.index = pd.PeriodIndex(pd.to_datetime(monthly.pop('Date')), freq='M')
    monthly = monthly.drop(columns=['Year', 'Month'])
    assert monthly.index.equals(pd.period_range(monthly.index[0], monthly.index[-1], freq='M'))
    weekly = pd.read_excel(SOURCE, 'Weekly_SST', skiprows=2).set_index('Week_ending')
    daily = pd.read_excel(SOURCE, 'Daily_OLR_ElNino', skiprows=2).set_index('Date')
    daily = daily.rename(columns={'OLR_ElNino_Wm2': 'OLREN'})
    for native in [weekly, daily]:
        assert native.index.is_unique
        means = native.groupby(pd.DatetimeIndex(native.index).to_period('M')).mean()
        np.testing.assert_allclose(means.reindex(monthly.index), monthly[means.columns],
                                   atol=1e-12, rtol=0, equal_nan=True)
    selected = monthly[monthly.index.month.isin([12, 1, 2])]
    groups = selected.groupby(selected.index.year + (selected.index.month == 12))
    winters = groups.mean().where(groups.count() == 3)
    # Independently address December, January and February to check boundary handling.
    direct = pd.DataFrame(index=winters.index, columns=monthly.columns, dtype=float)
    for year in direct.index:
        dates = [pd.Period(year=year - 1, month=12, freq='M'),
                 pd.Period(year=year, month=1, freq='M'),
                 pd.Period(year=year, month=2, freq='M')]
        direct.loc[year] = monthly.reindex(dates).sum(min_count=3) / 3
    np.testing.assert_allclose(winters, direct, atol=1e-13, rtol=0, equal_nan=True)
    saved = pd.read_excel(SOURCE, 'DJF_seasonal', skiprows=2).set_index('Winter')
    np.testing.assert_allclose(winters.loc[saved.index, saved.columns], saved,
                               atol=1e-13, rtol=0, equal_nan=True)
    overlap = winters[KEYS].dropna()
    assert overlap.index.tolist() == list(range(1982, 2023))
    assert np.isnan(winters.loc[1982, 'OLREN'])
    native = winters.loc[1983:2022, KEYS].copy()
    assert native.shape == (40, 8) and np.isfinite(native.to_numpy()).all()
    standardized = (native - native.mean()) / native.std(ddof=1)
    np.testing.assert_allclose(standardized.mean(), 0, atol=1e-13)
    np.testing.assert_allclose(standardized.std(ddof=1), 1, atol=1e-13)
    rows = []
    for key, name, unit in zip(KEYS, NAMES, UNITS):
        s = native[key]
        q1, median, q3 = np.quantile(s, [.25, .5, .75], method='linear')
        rows.append(dict(index=key, name=name, unit=unit, winters=len(s), mean=s.mean(),
                         sample_standard_deviation=s.std(ddof=1), minimum=s.min(),
                         first_quartile=q1, median=median, third_quartile=q3, maximum=s.max()))
    statistics = pd.DataFrame(rows)
    return winters, native, standardized, statistics


def plot(standardized, statistics):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'pdf.fonttype': 42, 'savefig.dpi': 300})
    fig, axes = plt.subplots(8, 1, figsize=(7, 8), sharex=True, sharey=True)
    fig.subplots_adjust(left=.10, right=.70, top=.88, bottom=.075, hspace=.82)
    fig.suptitle('Winter climate-index time series', y=.985, fontsize=13)
    fig.text(.5, .952, 'December–February, 1983–2022', ha='center', fontsize=9.5)
    fig.legend(handles=[Patch(facecolor='#B2182B', alpha=.25, label='Above period mean'),
                        Patch(facecolor='#2166AC', alpha=.25, label='Below period mean')],
               loc='upper center', bbox_to_anchor=(.49, .935), ncol=2, frameon=False,
               fontsize=8.5)
    years = standardized.index.to_numpy()
    for i, (ax, key, name, unit) in enumerate(zip(axes, KEYS, NAMES, UNITS)):
        values = standardized[key].to_numpy()
        stats = statistics.iloc[i]
        ax.fill_between(years, 0, values, where=values >= 0, interpolate=True,
                        color='#B2182B', alpha=.25, linewidth=0)
        ax.fill_between(years, 0, values, where=values <= 0, interpolate=True,
                        color='#2166AC', alpha=.25, linewidth=0)
        ax.plot(years, values, color='#262626', lw=.85, marker='o', markersize=1.7)
        ax.axhline(0, color='#888888', lw=.6, zorder=0)
        ax.set_title(f'({chr(97+i)}) {name}', loc='left', fontsize=9, pad=3)
        ax.text(1.07, .92, f'Mean: {stats["mean"]:+.3f}\n'
                f'Standard deviation: {stats.sample_standard_deviation:.3f}\n({unit})',
                transform=ax.transAxes, va='top', fontsize=8.5, linespacing=1.3)
        ax.set(ylim=(-3.3, 3.3), yticks=[-2, 0, 2], xlim=(1983, 2022),
               xticks=[1983, 1990, 2000, 2010, 2022])
        ax.tick_params(labelsize=8.5, length=2)
        ax.grid(axis='x', color='#DDDDDD', lw=.5)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_color('#BBBBBB')
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), years)
        np.testing.assert_array_equal(ax.lines[0].get_ydata(), values)
    axes[-1].set_xlabel('Winter year (January year)', fontsize=9)
    fig.text(.025, .48, 'Standardized departure (dimensionless)', rotation=90,
             va='center', fontsize=9)
    for extension in ['png', 'pdf']:
        fig.savefig(FIGURES / f'figure_s02_standardized_winter_time_series.{extension}',
                    facecolor='white')
    plt.close(fig)
