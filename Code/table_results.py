"""Map freshly calculated results to the numbered manuscript tables."""
from decimal import Decimal
import math
import pandas as pd
from table_definitions import TABLES
from winter_aggregation import KEYS, NAMES

LABELS = {'NINO12':'Niño 1+2', 'NINO3':'Niño 3', 'NINO34':'Niño 3.4',
          'NINO4':'Niño 4', 'OLREN':'El Niño OLR'}

def label(key):
    return LABELS.get(key, key)

def number(value, digits=3, signed=False):
    value = float(value)
    if round(value, digits)==0:
        value = 0.0
    return format(value, ('+' if signed else '')+f'.{digits}f').replace('-', '−')

def integer(value):
    return str(int(value))

def interval(low, high, signed=False, digits=3):
    return f'[{number(low,digits,signed)}, {number(high,digits,signed)}]'

def probability(value, main=False):
    if pd.isna(value):
        return '—'
    if value >= .001:
        return f'{value:.4f}'
    if main:
        return format(Decimal(f'{value:.3g}'), 'f')
    mantissa, exponent = f'{value:.2e}'.split('e')
    return mantissa+'×10'+str(int(exponent)).translate(str.maketrans('-0123456789','⁻⁰¹²³⁴⁵⁶⁷⁸⁹'))

def rows_for(key, study):
    """Return one row matrix per physical table, excluding its column headings."""
    f = study.frame
    n = number
    s = lambda x: number(x, signed=True)
    p = probability
    ci = lambda row, signed=False: interval(row.ci95_low, row.ci95_high, signed)
    if key in ('table_01', 'table_06'):
        return [v['rows'] for kind,v in TABLES[key]['blocks'] if kind=='table']
    if key=='table_02':
        source=study.inventory.set_index('index')
        names=dict(zip(KEYS,NAMES))
        names.update({k:label(k)+' SST anomaly' for k in ['NINO12','NINO3','NINO34','NINO4']})
        names.update(TNA='Tropical North Atlantic SST anomaly', OLREN='El Niño outgoing longwave radiation')
        rows=[]
        for k in ['AO','NAO','PNA','PDO','SOI','AMO','NINO34','TNA','NINO12','NINO3','NINO4','OLREN']:
            r=source.loc[k]
            coverage=pd.Period(r.first_numeric_month).strftime('%b %Y')+'–'+pd.Period(r.last_numeric_month).strftime('%b %Y')
            unit='Dimensionless' if k in ['AO','NAO','PNA','SOI'] else ('Provider scale' if k=='PDO' else ('°C (detrended)' if k=='AMO' else ('W m⁻²' if k=='OLREN' else '°C')))
            rows.append([label(k),names[k],r.native_cadence.capitalize(),coverage,unit])
        return [rows]
    if key=='table_03':
        df=f('winter_trend_results').set_index(['scope','index']); rows=[]
        for scope in ['own_record','common_1983_2022']:
            for k in ['AO','NAO','TNA','AMO','NINO12','NINO3','NINO34','NINO4']:
                r=df.loc[scope,k]
                rows.append([label(k),f'{r.start}–{r.end}',integer(r.n),s(r.tau_a),s(r.sen_per_decade)+'\n'+interval(r.scipy_ci95_low,r.scipy_ci95_high,True),p(r.p_tfpw,True),p(r.q_bh,True)])
        return [rows]
    if key=='table_04':
        a=f('winter_period_mean_comparisons').set_index('index')
        b=f('winter_period_difference_uncertainty').query('max_lag_years == 3').set_index('index')
        return [[[k,s(a.loc[k,'early_mean']),s(a.loc[k,'late_mean']),s(a.loc[k,'difference'])+'\n'+ci(b.loc[k],True),p(a.loc[k,'q_bh'],True),p(b.loc[k,'q_bh'],True)] for k in ['AO','NAO']]]
    if key=='table_05':
        a=f('winter_index_variance_inflation').set_index('index')
        b=f('winter_principal_component_weights').pivot(index='index',columns='component',values='eigenvector_weight')
        c=f('winter_index_represented_variance').set_index('index')
        names = dict(zip(KEYS,NAMES))
        names.update(NINO34='Niño 3.4', TNA='Tropical North Atlantic')
        return [[[names[k],n(a.loc[k,'variance_inflation_factor'],2),*[s(b.loc[k,i]) for i in [1,2,3]],n(100*c.loc[k,'correct_fraction_first_three'],2)] for k in KEYS]]
    if key=='table_s01':
        return [[[label(r.index),r.first_numeric_month+' to '+r.last_numeric_month,integer(r.available_months),integer(r.missing_months_within_record),integer(r.complete_calendar_years)] for r in study.inventory.itertuples() if r.index!='OLREN']]
    if key=='table_s02':
        cols=['mean','sample_standard_deviation','minimum','first_quartile','median','third_quartile','maximum']
        return [[[label(r['index']),*[n(r[c]) for c in cols]] for _,r in study.winter[3].iterrows()]]
    if key in ['table_s03','table_s04']:
        scope='own_record' if key=='table_s03' else 'common_1983_2022'
        rows=list(f('winter_trend_results').query('scope == @scope').itertuples())
        return [[[label(r.index),f'{r.start}–{r.end}',integer(r.n),s(r.tau_a),s(r.sen_per_decade),interval(r.scipy_ci95_low,r.scipy_ci95_high,True)] for r in rows],
                [[label(r.index),s(r.lag1_detrended),'Yes' if r.tfpw_applied else 'No',p(r.p),p(r.p_tfpw),p(r.q_bh)] for r in rows]]
    if key=='table_s05':
        return [[[label(r.index),f'{r.start}–{r.end}',integer(r.n),s(r.sen_per_decade)+'\n'+interval(r.scipy_ci95_low,r.scipy_ci95_high,True),s(r.lag1_detrended),'Yes' if r.tfpw_applied else 'No',p(r.p_tfpw)] for r in f('winter_trend_period_sensitivity').itertuples()]]
    if key=='table_s06':
        return [[[label(r.index),s(r.early_mean)+f' ({r.n_early})',s(r.late_mean)+f' ({r.n_late})',s(r.difference),p(r.welch_p),p(r.q_bh),p(r.mann_whitney_p)] for r in f('winter_period_mean_comparisons').itertuples()]]
    if key=='table_s07':
        return [[[label(r.index),integer(r.max_lag_years),integer(r.n),s(r.difference),n(r.se),ci(r,True),p(r.q_bh)] for r in f('winter_period_difference_uncertainty').itertuples()]]
    if key=='table_s08':
        return [[[label(r.pacific_index),integer(r.hac_lag_years),s(r.slope_difference_per_decade),ci(r,True),p(r.p),p(r.q_bh)] for r in f('atlantic_pacific_trend_contrasts').itertuples()]]
    if key=='table_s09':
        df=f('nino34_winter_variability_uncertainty'); r=df.iloc[0]
        return [[['1982–2004',integer(r.n_early),n(r.early_mean),n(r.early_sd)],['2005–2026',integer(r.n_late),n(r.late_mean),n(r.late_sd)]],
                [[integer(r.block_length_years),n(r.sd_ratio),ci(r)] for r in df.itertuples()]]
    if key=='table_s10':
        return [[[label(k),*[n(matrix.loc[k,j]) for j in KEYS]] for k in KEYS] for matrix in study.dependence[1:]]
    if key=='table_s11':
        df=f('winter_index_autocorrelations').set_index(['index','lag_winters'])
        return [[[label(k),*[n(df.loc[(k,i),'autocorrelation']) for i in lags]] for k in KEYS] for lags in [range(1,6),range(6,11)]]
    if key=='table_s12':
        return [[[label(r.index_1)+'–'+label(r.index_2),n(r.pearson),n(r.pearson_detrended),n(r.change_after_minus_before)] for r in f('winter_index_correlations').itertuples()]]
    if key=='table_s13':
        rows=[]
        for (a,b),g in f('winter_correlation_omission_sensitivity').groupby(['index_1','index_2'],sort=False):
            lo=g.loc[g.omission_correlation.idxmin()]; hi=g.loc[g.omission_correlation.idxmax()]
            rows.append([label(a)+'–'+label(b),n(lo.full_data_correlation),n(lo.omission_correlation),integer(lo.omitted_winter),n(hi.omission_correlation),integer(hi.omitted_winter)])
        return [rows]
    if key=='table_s14':
        return [[[label(r.index),n(r.variance_inflation_factor),n(r.auxiliary_r_squared),n(r.classical_standard_error_multiplier)] for r in f('winter_index_variance_inflation').itertuples()]]
    if key=='table_s15':
        return [[[integer(r.component),n(r.eigenvalue),n(r.variance_explained_percent,2),n(r.cumulative_percent,2)] for r in f('winter_principal_component_variance').itertuples()]]
    if key=='table_s16':
        df=f('winter_principal_component_weights').set_index(['index','component'])
        return [[[label(k),*[n(df.loc[(k,i),column]) for i in [1,2,3]]] for k in KEYS] for column in ['eigenvector_weight','index_score_correlation']]
    if key=='table_s17':
        return [[[label(r.index),n(100*r.correct_fraction_first_three,2),n(r.unweighted_squared_coefficients)] for r in f('winter_index_represented_variance').itertuples()]]
    if key=='table_s18':
        rows=list(f('winter_partial_correlations').itertuples())
        return [[[label(r.other_index),n(r.raw_correlation),n(r.partial_correlation),p(r.partial_classical_p),p(r.partial_classical_p_adjusted_six)] for r in rows],
                [[label(r.other_index),n(r.other_residual_lag1),n(r.pna_residual_lag1),n(r.residual_effective_count,2),p(r.partial_control_adjusted_effective_p),p(r.partial_control_adjusted_effective_p_adjusted_six),p(r.partial_recovered_effective_p)] for r in rows]]
    if key=='table_s19':
        return [[[label(r.other_index),integer(r.max_lag_winters),n(r.coefficient),n(r.standard_error),ci(r),p(r.p),p(r.adjusted_p_six_comparisons)] for r in f('winter_partial_regression_uncertainty').itertuples()]]
    if key=='table_s20':
        return [[[integer(r.winter_year),n(r.nino3_anomaly_degC),n(r.nino34_anomaly_degC),n(r.nino4_anomaly_degC),n(r.standardized_nino4_minus_nino3),r.winter_type] for r in study.events[1].itertuples()],
                [[r.reference_period.replace('-','–'),label(r.index),integer(r.winters),n(r.mean_degC),n(r.sample_standard_deviation_degC)] for r in f('el_nino_standardization_parameters').itertuples()]]
    if key=='table_s21':
        r=f('el_nino_fisher_inference').iloc[0]
        return [[[r.scenario.replace('-','–'),integer(r.earlier_central),integer(r.earlier_eastern),integer(r.later_central),integer(r.later_eastern),p(r.two_sided_p)] for r in f('el_nino_count_sensitivity').iloc[:3].itertuples()],
                [['Sample odds ratio',n(r.sample_odds_ratio,2)],['Conditional odds ratio',n(r.conditional_odds_ratio,2)],['Exact equal-tailed 95% interval',interval(r.exact_95_percent_lower,r.exact_95_percent_upper,digits=2)]]]
    if key=='table_s22':
        return [[[r.first_centre_month,r.last_centre_month,integer(r.consecutive_three_month_windows),r.selected_winter_years.replace(';',', ') or 'None'] for r in f('el_nino_continuous_warm_runs').itertuples()],
                [[r.scenario.split('_')[1],integer(r.earlier_central),integer(r.earlier_eastern),integer(r.later_central),integer(r.later_eastern),p(r.two_sided_p)] for r in f('el_nino_count_sensitivity').iloc[3:].itertuples()]]
    raise KeyError(key)
