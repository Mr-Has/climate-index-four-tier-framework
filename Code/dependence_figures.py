"""Reproduce Figures 6, 7 and S8-S11 without changing data or saved results.

Requires numpy, pandas, scipy, statsmodels, matplotlib and openpyxl. Source
aggregation and methods reuse the existing Figure S2 and Figure 5 scripts.
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

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT/'Figures'
KEYS = ['AO', 'NAO', 'PNA', 'PDO', 'SOI', 'AMO', 'NINO34', 'TNA']
SHORT = ['AO', 'NAO', 'PNA', 'PDO', 'SOI', 'AMO', 'Niño 3.4', 'TNA']
NAMES = ['Arctic Oscillation', 'North Atlantic Oscillation', 'Pacific–North American pattern',
         'Pacific Decadal Oscillation', 'Southern Oscillation Index',
         'Atlantic Multidecadal Oscillation', 'Niño 3.4', 'Tropical North Atlantic']
BLUE, RED, INK = '#2166AC', '#B2182B', '#262626'


def calculate():
    native = runpy.run_path(str(ROOT/'Code/winter_aggregation.py'))['calculate']()[1]
    methods = runpy.run_path(str(ROOT/'Code/index_dependence.py'))
    outputs, pearson, spearman = methods['calculate'](native)
    for name, frame in outputs.items():
        pd.testing.assert_frame_equal(pd.read_csv(ROOT/'Data'/name), frame,
                                      check_dtype=False, atol=1e-12, rtol=1e-10)
    z = (native-native.mean())/native.std(ddof=1)
    correlation = z.to_numpy().T@z.to_numpy()/39
    np.testing.assert_allclose(correlation, pearson, atol=1e-13)
    vif = outputs['winter_index_variance_inflation.csv'].variance_inflation_factor
    for i, value in enumerate(vif):
        minor = np.delete(np.delete(correlation, i, axis=0), i, axis=1)
        np.testing.assert_allclose(np.linalg.det(minor)/np.linalg.det(correlation), value, atol=1e-12)
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    np.testing.assert_allclose(eigenvalues, outputs['winter_principal_component_variance.csv'].eigenvalue, atol=1e-12)
    weights = outputs['winter_principal_component_weights.csv'].pivot(index='index', columns='component', values='eigenvector_weight').loc[KEYS].to_numpy()
    eigenvectors *= np.sign(np.sum(eigenvectors*weights, axis=0))
    np.testing.assert_allclose(eigenvectors, weights, atol=1e-12)
    reconstruction = z.to_numpy()@eigenvectors[:, :3]@eigenvectors[:, :3].T
    fraction = np.var(reconstruction, axis=0, ddof=1)
    np.testing.assert_allclose(fraction, outputs['winter_index_represented_variance.csv'].correct_fraction_first_three, atol=1e-12)
    for row in outputs['winter_partial_correlations.csv'].itertuples():
        # Precision matrix provides a separate check of the single-control partial correlation.
        selected = pearson.loc[[row.other_index, 'PNA', 'NINO34'], [row.other_index, 'PNA', 'NINO34']]
        precision = np.linalg.inv(selected)
        value = -precision[0, 1]/np.sqrt(precision[0, 0]*precision[1, 1])
        np.testing.assert_allclose(value, row.partial_correlation, atol=1e-12)
    return native, z, outputs, pearson


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#AAAAAA')
    ax.tick_params(labelsize=8, length=3)


def save(fig, stem):
    for ext in ['png', 'pdf']:
        fig.savefig(FIGURES /f'{stem}.{ext}', facecolor='white')
    plt.close(fig)


def component_variance(outputs):
    spectrum = outputs['winter_principal_component_variance.csv']
    represented = outputs['winter_index_represented_variance.csv'].correct_fraction_first_three.to_numpy()*100
    fig = plt.figure(figsize=(7, 6.7))
    fig.suptitle('Principal components of winter climate indices', y=.98, fontsize=12)
    ax = fig.add_axes([.12, .62, .83, .25])
    bars = ax.bar(spectrum.component, spectrum.variance_explained_percent, color=BLUE, alpha=.7, label='Individual component')
    line, = ax.plot(spectrum.component, spectrum.cumulative_percent, 'o-', color=RED, ms=3.5, lw=1.2, label='Cumulative')
    np.testing.assert_allclose([b.get_height() for b in bars], spectrum.variance_explained_percent)
    np.testing.assert_allclose(line.get_ydata(), spectrum.cumulative_percent)
    ax.annotate('83.64% in three components', (3, spectrum.cumulative_percent.iloc[2]),
                xytext=(3.55, 64), fontsize=8, arrowprops={'arrowstyle': '-', 'lw': .7, 'color': INK})
    ax.set(xticks=range(1, 9), ylim=(0, 105), ylabel='Total variance (%)', xlabel='Component number')
    ax.set_title('(a) Individual and cumulative variance', loc='left', fontsize=9)
    fig.legend([bars, line], ['Individual component', 'Cumulative'], frameon=False,
               fontsize=8, loc='upper center', bbox_to_anchor=(.53, .945), ncol=2)
    style(ax)
    ax = fig.add_axes([.34, .105, .61, .385])
    bars = ax.barh(range(8), represented, color=BLUE, alpha=.7, height=.62)
    np.testing.assert_allclose([b.get_width() for b in bars], represented)
    for y, value in enumerate(represented):ax.text(value+1, y, f'{value:.2f}', va='center', fontsize=8)
    ax.set(yticks=range(8), yticklabels=NAMES, ylim=(7.6, -.6), xlim=(0, 105),
           xticks=[0, 25, 50, 75, 100], xlabel='Index variance represented (%)')
    ax.set_title('(b) Representation by the first three components', loc='left', fontsize=9, pad=9)
    style(ax)
    save(fig, 'figure_06_principal_component_variance')


def scatter(ax, x, y, color=BLUE):
    points = ax.scatter(x, y, s=20, color=color, alpha=.75, edgecolors='white', linewidths=.35)
    np.testing.assert_allclose(points.get_offsets(), np.column_stack([x, y]), atol=0)
    fit = stats.linregress(x, y)
    line_x = np.array([min(x), max(x)])
    ax.plot(line_x, fit.intercept+fit.slope*line_x, color=RED, lw=1.2)
    style(ax)


def conditional_association(z, outputs):
    control = z.NINO34.to_numpy()
    residuals = {}
    for key in ['SOI', 'PNA']:
        fit = stats.linregress(control, z[key])
        residuals[key] = z[key].to_numpy()-fit.intercept-fit.slope*control
    row = outputs['winter_partial_correlations.csv'].query('other_index == "SOI"').iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(7, 4.5), sharex=True, sharey=True)
    fig.subplots_adjust(left=.11, right=.97, top=.76, bottom=.21, wspace=.23)
    fig.suptitle('Southern Oscillation and Pacific–North American relationships', y=.98, fontsize=10.5)
    fig.text(.5, .918, '40 winters, 1983–2022', ha='center', fontsize=9)
    for i, (ax, x, y, title) in enumerate(zip(axes, [z.SOI, residuals['SOI']], [z.PNA, residuals['PNA']],
                    ['(a) Unadjusted', '(b) Adjusted for Niño 3.4'])):
        scatter(ax, x, y)
        r = stats.pearsonr(x, y).statistic
        p = stats.pearsonr(x, y).pvalue if i == 0 else row.partial_classical_p
        np.testing.assert_allclose(r, row.raw_correlation if i == 0 else row.partial_correlation, atol=1e-13)
        annotation = f'r = {r:+.3f}; p = {p:.6f}' if i == 0 else f'Partial r = {r:+.3f}; p = {p:.4f}'
        ax.axhline(0, color='#BBBBBB', lw=.5, zorder=0);ax.axvline(0, color='#BBBBBB', lw=.5, zorder=0)
        ax.set(xlim=(-3.1, 2.6), ylim=(-2.5, 2.5),
               xlabel='Southern Oscillation Index\n(standardized units)' if i == 0 else 'Southern Oscillation Index residual\n(original standardized units)')
        ax.set_title(title+'\n'+annotation, loc='left', fontsize=8.5, pad=9)
    axes[0].set_ylabel('Pacific–North American pattern\n(standardized units)')
    axes[1].set_ylabel('Pattern residual\n(original standardized units)')
    save(fig, 'figure_07_soi_pna_partial_relationship')


def variance_inflation(outputs):
    values = outputs['winter_index_variance_inflation.csv'].variance_inflation_factor.to_numpy()
    fig, ax = plt.subplots(figsize=(7, 4.6))
    fig.subplots_adjust(left=.34, right=.97, top=.85, bottom=.14)
    bars = ax.barh(range(8), values, color=BLUE, alpha=.7, height=.65)
    np.testing.assert_allclose([b.get_width() for b in bars], values)
    for y, value in enumerate(values):ax.text(value+.1, y, f'{value:.2f}', va='center', fontsize=8)
    ax.axvline(5, color=INK, lw=1, ls='--')
    ax.set(yticks=range(8), yticklabels=NAMES, ylim=(7.6, -.6), xlim=(0, 8.1),
           xlabel='Variance inflation factor (dimensionless)')
    ax.set_title('Linear dependence among winter climate indices\n1983–2022', fontsize=11, pad=13)
    style(ax)
    save(fig, 'figure_s08_winter_index_variance_inflation')


def strongest_pairs(native):
    fig, axes = plt.subplots(1, 2, figsize=(7, 4.2))
    fig.subplots_adjust(left=.125, right=.97, top=.77, bottom=.23, wspace=.35)
    fig.suptitle('Strongest positive and negative winter correlations', y=.98, fontsize=11)
    for ax, first, second, title, xlabel, ylabel in zip(axes, ['AMO', 'NINO34'], ['TNA', 'SOI'],
            ['(a) Atlantic indices', '(b) Pacific indices'],
            ['Detrended Atlantic Multidecadal\nOscillation (°C)', 'Niño 3.4 temperature anomaly (°C)'],
            ['Tropical North Atlantic\ntemperature anomaly (°C)', 'Southern Oscillation Index\n(dimensionless)']):
        scatter(ax, native[first], native[second])
        r = stats.pearsonr(native[first], native[second]).statistic
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.set_title(f'{title}\nr = {r:+.3f}', loc='left', fontsize=9)
    save(fig, 'figure_s09_strongest_winter_index_correlations')


def detrending(outputs, pearson):
    after = pearson.copy()
    for row in outputs['winter_index_correlations.csv'].itertuples():
        after.loc[row.index_1, row.index_2] = after.loc[row.index_2, row.index_1] = row.pearson_detrended
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 4.8))
    fig.subplots_adjust(left=.08, right=.98, top=.8, bottom=.26, wspace=.27)
    fig.suptitle('Sensitivity of winter correlations to linear detrending', y=.98, fontsize=11)
    fig.text(.5, .925, '1983–2022 · 40 winters', ha='center', fontsize=9)
    for ax, matrix, title in zip(axes, [pearson, after], ['(a) Original indices', '(b) Additional linear detrending']):
        im = ax.imshow(matrix, vmin=-1, vmax=1, cmap='RdBu_r', interpolation='nearest')
        np.testing.assert_allclose(im.get_array(), matrix, atol=0)
        ax.set_title(title, fontsize=9, pad=9)
        ax.set_xticks(range(8), SHORT, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(8), SHORT, fontsize=8);ax.tick_params(length=0)
        ax.set_xticks(np.arange(8)-.5, minor=True);ax.set_yticks(np.arange(8)-.5, minor=True)
        ax.grid(which='minor', color='white', lw=.7);ax.tick_params(which='minor', length=0)
        for spine in ax.spines.values():spine.set_visible(False)
        for i in range(8):
            for j in range(8):
                value = matrix.iloc[i,j]
                ax.text(j, i, f'{0 if abs(value)<.005 else value:.2f}', ha='center', va='center',
                        fontsize=7.6, color='white' if abs(value)>.6 else INK)
    cax = fig.add_axes([.3, .085, .42, .025])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=[-1, -.5, 0, .5, 1])
    cb.set_label('Pearson correlation (dimensionless)', fontsize=8);cb.ax.tick_params(labelsize=8)
    save(fig, 'figure_s10_winter_correlations_after_detrending')


def omission(outputs):
    frame = outputs['winter_correlation_omission_sensitivity.csv']
    fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    fig.subplots_adjust(left=.12, right=.97, top=.82, bottom=.12, hspace=.55)
    fig.suptitle('Sensitivity to omitting one winter', y=.98, fontsize=12)
    fig.text(.5, .923, 'Dots: 39 retained winters · dashed line: all 40 winters', ha='center', fontsize=9)
    for ax, first, title in zip(axes, ['AMO', 'NINO34'],
                              ['(a) Detrended Atlantic Multidecadal Oscillation–Tropical North Atlantic',
                               '(b) Niño 3.4–Southern Oscillation Index']):
        rows = frame.query('index_1 == @first')
        points = ax.scatter(rows.omitted_winter, rows.omission_correlation, color=BLUE, s=16)
        np.testing.assert_allclose(points.get_offsets(), rows[['omitted_winter','omission_correlation']], atol=0)
        ax.axhline(rows.full_data_correlation.iloc[0], color=INK, ls='--', lw=1)
        lo, hi = rows.omission_correlation.min(), rows.omission_correlation.max()
        ax.set(ylim=(lo-.008, hi+.008), ylabel='Pearson correlation', xlim=(1982, 2023),
               xticks=[1983, 1990, 2000, 2010, 2022])
        ax.set_title(title, loc='left', fontsize=8.5)
        style(ax)
    axes[-1].set_xlabel('Omitted winter (January year)')
    save(fig, 'figure_s11_winter_correlation_omission_sensitivity')
