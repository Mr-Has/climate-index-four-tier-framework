"""Reproduce the complete-daily-coverage OLR comparison in Supplementary S1.2.

Run without options for a read-only check; --export saves two comparison CSVs
in Sensitivity. Original observations and original result files are unchanged.
"""
import sys
sys.dont_write_bytecode = True

import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from study_results import WORKBOOK_SHA256
from trend_analysis import TREND_FAMILY, adjust, mann_kendall, trend
from winter_aggregation import calculate as winter_means

ROOT = Path(__file__).resolve().parents[1]


def complete_coverage_trend(series):
    """Keep actual years in slopes and only consecutive-year residual pairs."""
    x = series.index.to_numpy(dtype=float)
    y = series.to_numpy(dtype=float)
    if len(y) < 4 or not np.isfinite(y).all() or not np.all(np.diff(x) > 0):
        raise ValueError('Expected finite observations in increasing winter years.')
    fit = stats.theilslopes(y, x, alpha=.95)
    i, j = np.triu_indices(len(y), 1)
    np.testing.assert_allclose(fit.slope, np.median((y[j]-y[i])/(x[j]-x[i])), atol=1e-14)
    residuals = y - fit.slope*(x-x[0])
    adjacent = np.diff(x) == 1
    pairs = int(adjacent.sum())
    if pairs < 3:
        raise ValueError('Too few consecutive-year pairs for the residual check.')
    correlation = np.corrcoef(residuals[:-1][adjacent], residuals[1:][adjacent])[0, 1]
    threshold = 1.96/np.sqrt(len(y))
    if not np.isfinite(correlation) or abs(correlation) > threshold:
        raise ValueError('The recorded no-pre-whitening sensitivity method no longer applies.')
    tau, probability = mann_kendall(y)
    return dict(start=int(x[0]), end=int(x[-1]), n=len(y), tau_a=float(tau),
                p=float(probability), p_tfpw=float(probability),
                sen_per_decade=float(10*fit.slope), scipy_ci95_low=float(10*fit.low_slope),
                scipy_ci95_high=float(10*fit.high_slope), lag1_detrended=float(correlation),
                lag1_pairs=pairs, filter_threshold=float(threshold),
                pair_count_threshold=float(1.96/np.sqrt(pairs)), tfpw_applied=False)


def calculate():
    source = ROOT/'Data/Climate_Indices_Data.xlsx'
    if sha256(source.read_bytes()).hexdigest() != WORKBOOK_SHA256:
        raise ValueError('The workbook differs from the reviewed data snapshot.')
    winters = winter_means()[0]
    daily = pd.read_excel(source, 'Daily_OLR_ElNino', skiprows=2)
    daily.index = pd.DatetimeIndex(pd.to_datetime(daily.pop('Date')))
    daily = daily['OLR_ElNino_Wm2']
    if not daily.index.is_unique or not daily.index.equals(daily.index.normalize()):
        raise ValueError('Daily dates must be unique calendar dates.')
    records = []
    for year, mean in winters.OLREN.dropna().items():
        dates = pd.date_range(f'{year-1}-12-01', f'{year}-03-01', inclusive='left')
        values = daily.reindex(dates)
        record = dict(winter_year=int(year), expected_days=len(dates),
                      observed_days=int(values.notna().sum()), missing_days=int(values.isna().sum()),
                      winter_mean_w_m2=float(mean), complete_daily_coverage=bool(values.notna().all()))
        for month, label in [(12, 'december'), (1, 'january'), (2, 'february')]:
            month_values = values[values.index.month == month]
            record[label+'_expected_days'] = len(month_values)
            record[label+'_observed_days'] = int(month_values.notna().sum())
        native_mean = values.groupby(values.index.to_period('M')).mean().mean()
        np.testing.assert_allclose(mean, native_mean, rtol=0, atol=1e-13)
        records.append(record)
    coverage = pd.DataFrame(records)
    excluded = coverage.loc[~coverage.complete_daily_coverage, 'winter_year'].tolist()
    if excluded != [1984, 1985, 1986] or coverage.missing_days.sum() != 54:
        raise ValueError('Coverage differs from the reviewed comparison.')
    saved = pd.read_csv(ROOT/'Data/winter_trend_results.csv', float_precision='round_trip')
    output = []
    for scope, frame in [('own_record', winters), ('common_1983_2022', winters.loc[1983:2022])]:
        original = [dict(index=k, **trend(frame[k])) for k in TREND_FAMILY]
        adjust(original, 'p_tfpw')
        reference = saved.loc[saved.scope == scope].set_index('index')
        for row in original:
            for field in ['sen_per_decade', 'p', 'p_tfpw', 'q_bh']:
                np.testing.assert_allclose(row[field], reference.loc[row['index'], field], rtol=1e-12, atol=1e-13)
        complete = [dict(row) for row in original]
        olr = next(row for row in complete if row['index'] == 'OLREN')
        series = frame.OLREN.dropna().drop(index=excluded, errors='ignore')
        olr.update(complete_coverage_trend(series))
        adjust(complete, 'p_tfpw')
        for condition, family in [('original', original), ('complete_daily_coverage', complete)]:
            for row in family:
                output.append(dict(scope=scope, coverage=condition, **row,
                                   significant_at_005=bool(row['q_bh'] < .05)))
        if [r['q_bh'] < .05 for r in original] != [r['q_bh'] < .05 for r in complete]:
            raise AssertionError('A family decision differs from the reviewed result.')
    if sha256(source.read_bytes()).hexdigest() != WORKBOOK_SHA256:
        raise AssertionError('Source workbook changed during calculation.')
    return {'olr_winter_daily_coverage.csv': coverage,
            'olr_coverage_trend_family_comparison.csv': pd.DataFrame(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export', action='store_true', help='Save comparison CSVs in Sensitivity')
    args = parser.parse_args()
    protected = {p: sha256(p.read_bytes()).hexdigest() for p in (ROOT/'Data').iterdir() if p.is_file()}
    frames = calculate()
    comparison = frames['olr_coverage_trend_family_comparison.csv']
    print(comparison.loc[comparison['index'] == 'OLREN',
          ['scope', 'coverage', 'n', 'sen_per_decade', 'p_tfpw', 'q_bh']].to_string(index=False))
    if any(sha256(p.read_bytes()).hexdigest() != digest for p, digest in protected.items()):
        raise AssertionError('An archived input or result changed.')
    if args.export:
        destination = ROOT/'Sensitivity'
        destination.mkdir(exist_ok=True)
        for name, frame in frames.items():
            frame.to_csv(destination/name, index=False, encoding='utf-8-sig', float_format='%.17g')
        print('Saved two comparison CSVs in Sensitivity.')
    print('Original inputs and result files unchanged. No adjusted significance decisions changed.')


if __name__ == '__main__':
    main()
