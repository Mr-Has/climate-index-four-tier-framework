"""Recalculate Section 4.4, Tables S10-S19 and the correlation figure.

Input observations are unchanged. Requires numpy, pandas, scipy, statsmodels,
matplotlib and openpyxl. Paths are relative to this file.
"""
from pathlib import Path
from hashlib import sha256
import runpy
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT/'Figures'
KEYS = ['AO', 'NAO', 'PNA', 'PDO', 'SOI', 'AMO', 'NINO34', 'TNA']
OTHER = ['PDO', 'AO', 'NAO', 'TNA', 'AMO', 'SOI']


def probability(r, degrees):
    return 2*stats.t.sf(abs(r)*np.sqrt(degrees/(1-r*r)), degrees)


def effective_count(x, y):
    a = np.corrcoef(x[:-1], x[1:])[0, 1]
    b = np.corrcoef(y[:-1], y[1:])[0, 1]
    return min(len(x), len(x)*(1-a*b)/(1+a*b)), a, b


def calculate(native):
    assert native.shape == (40, 8) and list(native) == KEYS
    assert native.index.tolist() == list(range(1983, 2023))
    assert np.isfinite(native.to_numpy()).all()
    z = (native-native.mean())/native.std(ddof=1)
    a = z.to_numpy()
    n = len(a)
    pearson, spearman = native.corr(), native.corr(method='spearman')
    np.testing.assert_allclose(pearson, np.corrcoef(a.T), atol=1e-13)
    np.testing.assert_allclose(spearman, stats.spearmanr(native).statistic, atol=1e-13)
    time = np.column_stack([np.ones(n), np.arange(n)])
    detrended = a-time@np.linalg.lstsq(time, a, rcond=None)[0]
    detrended_correlation = np.corrcoef(detrended.T)
    pairs = []
    for i in range(8):
        for j in range(i+1, 8):
            pairs.append(dict(index_1=KEYS[i], index_2=KEYS[j], pearson=pearson.iloc[i, j],
                spearman=spearman.iloc[i, j], pearson_detrended=detrended_correlation[i, j],
                change_after_minus_before=detrended_correlation[i, j]-pearson.iloc[i, j]))
    autocorrelation = []
    for key in KEYS:
        values = native[key].to_numpy()-native[key].mean()
        for lag in range(1, 11):
            autocorrelation.append(dict(index=key, lag_winters=lag, contributing_pairs=n-lag,
                autocorrelation=values[lag:]@values[:-lag]/(values@values)))
    # The original omission analysis examined the strongest positive and negative pairs.
    selected = [('AMO', 'TNA'), ('NINO34', 'SOI')]
    omission = []
    for first, second in selected:
        for year in native.index:
            subset = native.drop(index=year)
            value = stats.pearsonr(subset[first], subset[second]).statistic
            omission.append(dict(index_1=first, index_2=second, omitted_winter=year,
                retained_winters=len(subset), full_data_correlation=pearson.loc[first, second],
                omission_correlation=value, change_from_full=value-pearson.loc[first, second]))
    assert np.linalg.matrix_rank(a) == 8
    vif = []
    for i, key in enumerate(KEYS):
        design = np.column_stack([np.ones(n), np.delete(a, i, axis=1)])
        residual = a[:, i]-design@np.linalg.lstsq(design, a[:, i], rcond=None)[0]
        value = np.sum(a[:, i]**2)/np.sum(residual**2)
        np.testing.assert_allclose(value, np.linalg.inv(pearson)[i, i], atol=1e-12)
        vif.append(dict(index=key, variance_inflation_factor=value, auxiliary_r_squared=1-1/value,
                        classical_standard_error_multiplier=np.sqrt(value)))
    _, singular, vt = np.linalg.svd(a, full_matrices=False)
    weights = vt.T.copy()
    eigenvalues = singular**2/(n-1)
    for j in range(8):
        # Retain the paper's orientation for the first three components.
        direction = weights[0, j]*[-1, -1, 1][j] if j < 3 else weights[np.argmax(abs(weights[:, j])), j]
        weights[:, j] *= 1 if direction >= 0 else -1
    scores = a@weights
    correlations = weights*np.sqrt(eigenvalues)
    independent_values = np.linalg.eigvalsh(pearson)[::-1]
    np.testing.assert_allclose(eigenvalues, independent_values, atol=1e-12)
    np.testing.assert_allclose(scores@weights.T, a, atol=1e-12)
    np.testing.assert_allclose(np.corrcoef(a.T, scores.T)[:8, 8:], correlations, atol=1e-12)
    explained = eigenvalues/eigenvalues.sum()*100
    spectrum = pd.DataFrame(dict(component=np.arange(1, 9), eigenvalue=eigenvalues,
        variance_explained_percent=explained, cumulative_percent=np.cumsum(explained)))
    loadings = pd.DataFrame([dict(index=key, component=j+1, eigenvector_weight=weights[i, j],
        index_score_correlation=correlations[i, j]) for i, key in enumerate(KEYS) for j in range(8)])
    fraction = (correlations[:, :3]**2).sum(axis=1)
    reconstructed = scores[:, :3]@weights[:, :3].T
    np.testing.assert_allclose(fraction, 1-np.sum((a-reconstructed)**2, axis=0)/np.sum(a*a, axis=0), atol=1e-12)
    represented = pd.DataFrame(dict(index=KEYS, correct_fraction_first_three=fraction,
        unweighted_squared_coefficients=(weights[:, :3]**2).sum(axis=1)))
    control = np.column_stack([np.ones(n), z.NINO34])
    y = z.PNA.to_numpy()
    ry = y-control@np.linalg.lstsq(control, y, rcond=None)[0]
    partial, robust_rows = [], []
    for key in OTHER:
        x = z[key].to_numpy()
        rx = x-control@np.linalg.lstsq(control, x, rcond=None)[0]
        raw, r = np.corrcoef(x, y)[0, 1], np.corrcoef(rx, ry)[0, 1]
        rxc, ryc = pearson.loc[key, 'NINO34'], pearson.loc['PNA', 'NINO34']
        np.testing.assert_allclose(r, (raw-rxc*ryc)/np.sqrt((1-rxc*rxc)*(1-ryc*ryc)), atol=1e-12)
        effective, lag_x, lag_y = effective_count(rx, ry)
        design = np.column_stack([np.ones(n), x, z.NINO34])
        fit = sm.OLS(y, design).fit()
        classical = probability(r, n-3)
        np.testing.assert_allclose(classical, fit.pvalues[1], atol=1e-12)
        partial.append(dict(other_index=key, raw_correlation=raw, partial_correlation=r,
            partial_classical_p=classical, residual_effective_count=effective,
            other_residual_lag1=lag_x, pna_residual_lag1=lag_y,
            partial_control_adjusted_effective_p=probability(r, effective-3),
            partial_recovered_effective_p=probability(r, effective-2), regression_coefficient=fit.params[1]))
        for lag in [1, 3, 5]:
            robust = fit.get_robustcov_results(cov_type='HAC', maxlags=lag, kernel='bartlett',
                                               use_correction=True, use_t=True)
            score = design*fit.resid[:, None]
            middle = score.T@score
            for h in range(1, lag+1):
                cross = score[h:].T@score[:-h]
                middle += (1-h/(lag+1))*(cross+cross.T)
            inverse = np.linalg.inv(design.T@design)
            np.testing.assert_allclose(inverse@middle@inverse*n/(n-3), robust.cov_params(), atol=1e-12)
            lo, hi = robust.conf_int()[1]
            robust_rows.append(dict(other_index=key, max_lag_winters=lag, coefficient=robust.params[1],
                standard_error=robust.bse[1], ci95_low=lo, ci95_high=hi, p=robust.pvalues[1]))
    partial, robust = pd.DataFrame(partial), pd.DataFrame(robust_rows)
    for field in ['partial_classical_p', 'partial_control_adjusted_effective_p']:
        partial[field+'_adjusted_six'] = stats.false_discovery_control(partial[field], method='bh')
    for lag in [1, 3, 5]:
        select = robust.max_lag_winters == lag
        robust.loc[select, 'adjusted_p_six_comparisons'] = stats.false_discovery_control(robust.loc[select, 'p'], method='bh')
    return {
        'winter_index_correlations.csv': pd.DataFrame(pairs),
        'winter_index_autocorrelations.csv': pd.DataFrame(autocorrelation),
        'winter_correlation_omission_sensitivity.csv': pd.DataFrame(omission),
        'winter_index_variance_inflation.csv': pd.DataFrame(vif),
        'winter_principal_component_variance.csv': spectrum,
        'winter_principal_component_weights.csv': loadings,
        'winter_index_represented_variance.csv': represented,
        'winter_partial_correlations.csv': partial,
        'winter_partial_regression_uncertainty.csv': robust,
    }, pearson, spearman


def plot(pearson, spearman):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42,
                         'savefig.dpi': 400})
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 4.8))
    fig.subplots_adjust(left=.08, right=.98, bottom=.26, top=.80, wspace=.27)
    fig.suptitle('Relationships among winter climate indices', y=.98, fontsize=12)
    fig.text(.5, .92, 'December–February, 1983–2022 · 40 winters', ha='center', fontsize=9)
    labels = ['AO', 'NAO', 'PNA', 'PDO', 'SOI', 'AMO', 'Niño 3.4', 'TNA']
    for ax, matrix, title in zip(axes, [pearson, spearman], ['(a) Pearson correlation', '(b) Spearman correlation']):
        im = ax.imshow(matrix, vmin=-1, vmax=1, cmap='RdBu_r', interpolation='nearest')
        ax.set_title(title, fontsize=10, pad=9)
        ax.set_xticks(range(8), labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(8), labels, fontsize=8)
        ax.tick_params(length=0)
        ax.set_xticks(np.arange(8)-.5, minor=True); ax.set_yticks(np.arange(8)-.5, minor=True)
        ax.grid(which='minor', color='white', linewidth=.7); ax.tick_params(which='minor', length=0)
        for spine in ax.spines.values(): spine.set_visible(False)
        for i in range(8):
            for j in range(8):
                value = matrix.iloc[i, j]
                ax.text(j, i, f'{0 if abs(value)<.005 else value:.2f}', ha='center', va='center',
                        fontsize=7.6, color='white' if abs(value)>.6 else '#262626')
        np.testing.assert_allclose(im.get_array(), matrix, atol=0)
    color_ax = fig.add_axes([.30, .085, .42, .025])
    cb = fig.colorbar(im, cax=color_ax, orientation='horizontal', ticks=[-1, -.5, 0, .5, 1])
    cb.set_label('Correlation coefficient (dimensionless)', fontsize=8)
    cb.ax.tick_params(labelsize=8)
    for extension in ['png', 'pdf']:
        fig.savefig(FIGURES /f'figure_05_climate_index_correlations.{extension}', facecolor='white')
    plt.close(fig)
