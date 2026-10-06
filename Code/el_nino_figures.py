"""Verify archived El Nino calculations and draw Figures 8 and S12."""
from hashlib import sha256
from pathlib import Path
import runpy
import sys

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT/'Figures'


def verify(outputs):
    for name, frame in outputs.items():
        text_columns = {key: str for key in frame if frame[key].dtype == object}
        saved = pd.read_csv(ROOT/'Data'/name, float_precision='round_trip', dtype=text_columns)
        assert list(saved.columns) == list(frame.columns) and saved.shape == frame.shape
        for key in frame:
            if pd.api.types.is_numeric_dtype(frame[key]):
                np.testing.assert_allclose(saved[key], frame[key], atol=1e-13, rtol=1e-13)
            else:
                assert saved[key].fillna('').astype(str).tolist() == frame[key].fillna('').astype(str).tolist()
    # Rebuild monthly and winter means directly from workbook records.
    weekly = pd.read_excel(ROOT/'Data/Climate_Indices_Data.xlsx', 'Weekly_SST', skiprows=2)
    dates = pd.to_datetime(weekly.Week_ending)
    source = outputs['el_nino_winter_classification.csv']
    for index, column in [('NINO3', 'nino3_anomaly_degC'),
                          ('NINO34', 'nino34_anomaly_degC'), ('NINO4', 'nino4_anomaly_degC')]:
        rebuilt = []
        for year in range(1982, 2027):
            means = []
            for yr, month in [(year-1, 12), (year, 1), (year, 2)]:
                values = weekly.loc[(dates.dt.year == yr) & (dates.dt.month == month), index].to_numpy()
                assert len(values) in [4, 5] and np.isfinite(values).all()
                means.append(sum(values)/len(values))
            rebuilt.append(sum(means)/3)
        np.testing.assert_allclose(source[column], rebuilt, atol=1e-14, rtol=1e-14)
    assert source.loc[source.selected_el_nino_winter, 'winter_year'].tolist() == [
        1983, 1987, 1992, 1995, 1998, 2003, 2007, 2010, 2015, 2016, 2019, 2024]
    # Rebuild contiguous warm runs without the calculation function's group identifiers.
    windows = outputs['el_nino_three_month_windows.csv']
    spans, start = [], None
    for i, value in enumerate(windows.nino34_three_month_anomaly_degC):
        if value >= .5 and start is None:
            start = i
        if start is not None and (value < .5 or i == len(windows)-1):
            end = i-1 if value < .5 else i
            spans.append((windows.centre_month.iloc[start], windows.centre_month.iloc[end], end-start+1))
            start = None
    runs = outputs['el_nino_continuous_warm_runs.csv']
    assert spans == list(runs[['first_centre_month', 'last_centre_month',
                               'consecutive_three_month_windows']].itertuples(index=False, name=None))


def plot_sensitivity(outputs, selected):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
        'pdf.fonttype': 42, 'savefig.dpi': 300,
        'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 1, figsize=(7, 6.5),
                             gridspec_kw={'height_ratios': [4, 1]})
    fig.subplots_adjust(left=.14, right=.97, top=.82, bottom=.09, hspace=.72)
    fig.suptitle('El Niño classification sensitivity and uncertainty', y=.978, fontsize=12)
    records = outputs['el_nino_standardization_sensitivity.csv']
    styles = [('1982-2026', '#2166AC', 'o', -.2),
              ('1991-2020', '#B2182B', 's', 0),
              ('1982-2004', '#666666', '^', .2)]
    for period, color, marker, offset in styles:
        frame = records.loc[records.reference_period == period]
        assert frame.winter_year.tolist() == selected.winter_year.tolist()
        x = frame.standardized_nino4_minus_nino3.to_numpy()
        y = np.arange(len(frame)) + offset
        points = axes[0].scatter(x, y, color=color, marker=marker, s=22,
                                label=period.replace('-', '–'), zorder=3)
        np.testing.assert_allclose(points.get_offsets(), np.column_stack([x, y]))
    axes[0].axvline(0, color='.35', linestyle='--', linewidth=.8)
    axes[0].set(yticks=np.arange(len(selected)), yticklabels=selected.winter_year,
                ylim=(11.6, -.6), xlim=(-2.4, 1), ylabel='Winter (January year)',
                xlabel='Standardized Niño 4 minus Niño 3 (dimensionless)')
    axes[0].set_title('(a) Contrast under three standardization periods', loc='left', fontsize=10)
    axes[0].grid(axis='x', color='.9', linewidth=.5)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center',
               bbox_to_anchor=(.55, .94), ncol=3, frameon=False,
               title='Standardization period', fontsize=9, title_fontsize=9)
    result = outputs['el_nino_fisher_inference.csv'].iloc[0]
    estimate = result.conditional_odds_ratio
    low, high = result.exact_95_percent_lower, result.exact_95_percent_upper
    axes[1].plot([low, high], [0, 0], color='#2166AC', linewidth=1.4)
    axes[1].scatter([estimate], [0], color='#2166AC', s=28, zorder=3)
    axes[1].vlines([low, high], -.09, .09, color='#2166AC', linewidth=1.1)
    axes[1].axvline(1, color='.35', linestyle='--', linewidth=.8)
    axes[1].set(xscale='log', xlim=(.1, 1000), ylim=(-.4, .5), yticks=[],
                xlabel='Later-to-earlier odds ratio (logarithmic scale)',
                xticks=[.1, 1, 10, 100, 1000], xticklabels=['0.1', '1', '10', '100', '1000'])
    axes[1].spines['left'].set_visible(False)
    axes[1].set_title('(b) Primary classification: conditional estimate and exact 95% interval',
                      loc='left', fontsize=10, pad=20)
    axes[1].text(.5, 1.01, f'{estimate:.2f} [{low:.2f}, {high:.2f}]; two-sided Fisher p = {result.two_sided_p:.4f}',
                 transform=axes[1].transAxes, ha='center', fontsize=9)
    np.testing.assert_allclose(axes[1].lines[0].get_xdata(), [low, high])
    for extension in ['png', 'pdf']:
        fig.savefig(FIGURES /f'figure_s12_el_nino_classification_sensitivity.{extension}', facecolor='white')
    plt.close(fig)
