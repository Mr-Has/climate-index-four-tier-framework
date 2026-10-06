"""Recalculate the results in supplementary Tables S3-S9 from the source workbook.

Run from any directory with Python, numpy, pandas, scipy, statsmodels,
matplotlib and openpyxl installed. No source observations are overwritten.
"""
from pathlib import Path
from hashlib import sha256
import runpy

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'Data' / 'Climate_Indices_Data.xlsx'
TREND_FAMILY = ['AO', 'NAO', 'PNA', 'PDO', 'SOI', 'AMO', 'NINO34',
                'NINO12', 'NINO4', 'TNA', 'OLREN']
PERIOD_FAMILY = ['AO', 'NAO', 'PNA', 'PDO', 'SOI']
PACIFIC = ['NINO3', 'NINO34', 'NINO4']


def mann_kendall(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    score = sum(np.sign(values[j] - values[:j]).sum() for j in range(1, n))
    ties = np.unique(values, return_counts=True)[1]
    variance = (n*(n-1)*(2*n+5) - np.sum(ties*(ties-1)*(2*ties+5))) / 18
    z = (score - np.sign(score)) / np.sqrt(variance)
    return score / (n*(n-1)/2), 2 * stats.norm.sf(abs(z))


def trend(series):
    s = series.dropna()
    x, y = s.index.to_numpy(float), s.to_numpy(float)
    assert np.all(np.diff(x) == 1), 'Serial-dependence calculation requires consecutive winters.'
    fit = stats.theilslopes(y, x, alpha=.95)
    residuals = y - fit.slope * (x-x[0])
    correlation = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
    apply = abs(correlation) > 1.96 / np.sqrt(len(y))
    tau, nominal = mann_kendall(y)
    filtered = residuals[1:] - correlation*residuals[:-1] + fit.slope*(x[1:]-x[0])
    i, j = np.triu_indices(len(y), 1)
    np.testing.assert_allclose(fit.slope, np.median((y[j]-y[i])/(x[j]-x[i])), atol=1e-14)
    return dict(start=int(x[0]), end=int(x[-1]), n=len(y), tau_a=float(tau), p=float(nominal),
                sen_per_decade=float(10*fit.slope), scipy_ci95_low=float(10*fit.low_slope),
                scipy_ci95_high=float(10*fit.high_slope), lag1_detrended=float(correlation),
                tfpw_applied=bool(apply), p_tfpw=float(mann_kendall(filtered)[1] if apply else nominal))


def adjust(rows, probability):
    values = np.array([r[probability] for r in rows])
    adjusted = stats.false_discovery_control(values, method='bh')
    order = np.argsort(values)
    manual = np.minimum.accumulate((values[order]*len(values)/np.arange(1, len(values)+1))[::-1])[::-1]
    np.testing.assert_allclose(adjusted[order], np.minimum(manual, 1), atol=1e-14)
    for row, value in zip(rows, adjusted):
        row['q_bh'] = float(value)


def hac_regression(values, predictor, lag):
    design = sm.add_constant(np.asarray(predictor, float))
    fit = sm.OLS(np.asarray(values, float), design).fit(cov_type='HAC',
        cov_kwds={'maxlags': lag, 'kernel': 'bartlett', 'use_correction': True}, use_t=True)
    scores = design*fit.resid[:, None]
    middle = scores.T @ scores
    for h in range(1, lag+1):
        cross = scores[h:].T @ scores[:-h]
        middle += (1-h/(lag+1))*(cross+cross.T)
    inverse = np.linalg.inv(design.T @ design)
    manual_covariance = inverse @ middle @ inverse * len(values)/(len(values)-2)
    np.testing.assert_allclose(manual_covariance, fit.cov_params(), atol=1e-12, rtol=1e-10)
    return fit


def circular_samples(values, length, rng, draws=10000):
    values = np.asarray(values)
    starts = rng.integers(0, len(values), size=(draws, int(np.ceil(len(values)/length))))
    positions = (starts[:, :, None] + np.arange(length)) % len(values)
    return values[positions.reshape(draws, -1)[:, :len(values)]]


def calculate(winters):
    trends = []
    for scope, frame in [('own_record', winters), ('common_1983_2022', winters.loc[1983:2022])]:
        family = [dict(index=k, scope=scope, **trend(frame[k]), in_original_bh_family=True)
                  for k in TREND_FAMILY]
        adjust(family, 'p_tfpw')
        family.append(dict(index='NINO3', scope=scope, **trend(frame.NINO3),
                           q_bh=np.nan, in_original_bh_family=False))
        trends.extend(family)
    sensitivity = []
    for key in ['AO', 'NAO']:
        for start, end in [(1951, 2026), (1971, 2026), (1983, 2026), (1983, 2022)]:
            sensitivity.append(dict(index=key, period=f'{start}\u2013{end}', **trend(winters.loc[start:end, key])))
    periods = []
    for key in PERIOD_FAMILY:
        early, late = [winters.loc[a:b, key].dropna() for a, b in [(1951, 1988), (1989, 2026)]]
        # Welch's statistic and degrees of freedom checked independently.
        variance_early, variance_late = early.var(ddof=1)/len(early), late.var(ddof=1)/len(late)
        statistic = (early.mean()-late.mean())/np.sqrt(variance_early+variance_late)
        degrees = (variance_early+variance_late)**2 / (variance_early**2/(len(early)-1)+variance_late**2/(len(late)-1))
        probability = stats.ttest_ind(early, late, equal_var=False).pvalue
        np.testing.assert_allclose(probability, 2*stats.t.sf(abs(statistic), degrees), atol=1e-14)
        periods.append(dict(index=key, n_early=len(early), n_late=len(late), early_mean=early.mean(),
            late_mean=late.mean(), difference=late.mean()-early.mean(), welch_p=probability,
            mann_whitney_p=stats.mannwhitneyu(early, late, alternative='two-sided', method='asymptotic',
                                            use_continuity=True).pvalue))
    adjust(periods, 'welch_p')
    differences, contrasts = [], []
    for lag in [1, 3, 5]:
        family = []
        for key in PERIOD_FAMILY:
            s = winters.loc[1951:2026, key].dropna()
            assert np.all(np.diff(s.index) == 1)
            fit = hac_regression(s, (s.index >= 1989).astype(float), lag)
            low, high = fit.conf_int()[1]
            expected = next(r['difference'] for r in periods if r['index'] == key)
            np.testing.assert_allclose(fit.params[1], expected, atol=1e-14)
            family.append(dict(index=key, max_lag_years=lag, n=len(s), difference=fit.params[1],
                se=fit.bse[1], ci95_low=low, ci95_high=high, p=fit.pvalues[1]))
        adjust(family, 'p')
        differences.extend(family)
        family = []
        s = winters.loc[1982:2026]
        time = (s.index.to_numpy()-1982)/10
        for key in PACIFIC:
            fit = hac_regression(s.TNA-s[key], time, lag)
            low, high = fit.conf_int()[1]
            separate = stats.linregress(time, s.TNA).slope - stats.linregress(time, s[key]).slope
            np.testing.assert_allclose(fit.params[1], separate, atol=1e-14)
            family.append(dict(pacific_index=key, hac_lag_years=lag, n=len(s), start=1982, end=2026,
                slope_difference_per_decade=fit.params[1], ci95_low=low, ci95_high=high, p=fit.pvalues[1]))
        adjust(family, 'p')
        contrasts.extend(family)
    early, late = winters.loc[1982:2004, 'NINO34'], winters.loc[2005:2026, 'NINO34']
    assert len(early) == 23 and len(late) == 22 and early.notna().all() and late.notna().all()
    variability = []
    for length in [2, 3, 5]:
        seed = 20260922 + length
        rng = np.random.default_rng(seed)
        a, b = circular_samples(early, length, rng), circular_samples(late, length, rng)
        low, high = np.quantile(b.std(axis=1, ddof=1)/a.std(axis=1, ddof=1), [.025, .975], method='linear')
        variability.append(dict(block_length_years=length, draws=10000, seed=seed,
            sd_ratio=late.std(ddof=1)/early.std(ddof=1), ci95_low=low, ci95_high=high,
            n_early=len(early), n_late=len(late), early_mean=early.mean(), late_mean=late.mean(),
            early_sd=early.std(ddof=1), late_sd=late.std(ddof=1)))
    return {
        'winter_trend_results.csv': pd.DataFrame(trends),
        'winter_trend_period_sensitivity.csv': pd.DataFrame(sensitivity),
        'winter_period_mean_comparisons.csv': pd.DataFrame(periods),
        'winter_period_difference_uncertainty.csv': pd.DataFrame(differences),
        'atlantic_pacific_trend_contrasts.csv': pd.DataFrame(contrasts),
        'nino34_winter_variability_uncertainty.csv': pd.DataFrame(variability),
    }
