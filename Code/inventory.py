"""Reproduce the archived record inventory and supplementary availability figure."""
from pathlib import Path
import hashlib
import textwrap

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT/'Figures'
SOURCE = ROOT / 'Data' / 'Climate_Indices_Data.xlsx'
NAMES = {
    'AO': 'Arctic Oscillation',
    'NAO': 'North Atlantic Oscillation',
    'PNA': 'Pacific–North American pattern',
    'SOI': 'Southern Oscillation Index',
    'PDO': 'Pacific Decadal Oscillation',
    'AMO': 'Atlantic Multidecadal Oscillation',
    'NINO12': 'Niño 1+2 temperature anomaly',
    'NINO3': 'Niño 3 temperature anomaly',
    'NINO34': 'Niño 3.4 temperature anomaly',
    'NINO4': 'Niño 4 temperature anomaly',
    'TNA': 'Tropical North Atlantic temperature anomaly',
    'OLREN': 'El Niño outgoing longwave radiation index',
}
WEEKLY = ['NINO12', 'NINO3', 'NINO34', 'NINO4', 'TNA']


def calculate():
    monthly = pd.read_excel(SOURCE, 'Monthly', skiprows=2)
    months = pd.PeriodIndex(pd.to_datetime(monthly.Date), freq='M')
    monthly.index = months
    assert months.is_unique and months.equals(pd.period_range(months[0], months[-1], freq='M'))
    weekly = pd.read_excel(SOURCE, 'Weekly_SST', skiprows=2).set_index('Week_ending')
    weekly.index = pd.DatetimeIndex(weekly.index)
    assert weekly.index.equals(pd.date_range(weekly.index[0], weekly.index[-1], freq='7D'))
    assert weekly[WEEKLY].notna().all().all()
    daily = pd.read_excel(SOURCE, 'Daily_OLR_ElNino', skiprows=2).set_index('Date').iloc[:, 0]
    daily.index = pd.DatetimeIndex(daily.index)
    assert daily.index.equals(pd.date_range(daily.index[0], daily.index[-1]))
    metadata = pd.read_excel(SOURCE, 'Metadata', skiprows=2).set_index('Index')
    rows = []
    for key, name in NAMES.items():
        native = weekly[key] if key in WEEKLY else daily if key == 'OLREN' else monthly[key]
        valid = native.dropna()
        ms = monthly[key].dropna()
        interior = monthly.loc[ms.index[0]:ms.index[-1], key]
        assert interior.notna().all()
        annual_counts = ms.groupby(ms.index.year).size()
        if key in WEEKLY or key == 'OLREN':
            aggregated = native.groupby(native.index.to_period('M')).mean()
            np.testing.assert_allclose(aggregated.reindex(months), monthly[key], atol=1e-12, rtol=0, equal_nan=True)
        missing_archive = int(metadata.loc[key, 'Missing'])
        assert len(valid) + missing_archive == int(metadata.loc[key, 'N'])
        first = pd.Timestamp(monthly.loc[valid.index[0], 'Date']) if isinstance(valid.index, pd.PeriodIndex) else valid.index[0]
        last = pd.Timestamp(monthly.loc[valid.index[-1], 'Date']) if isinstance(valid.index, pd.PeriodIndex) else valid.index[-1]
        rows.append(dict(index=key, full_name=name, native_cadence='weekly' if key in WEEKLY else 'daily' if key == 'OLREN' else 'monthly',
            first_valid_date=first.strftime('%Y-%m-%d'), last_valid_date=last.strftime('%Y-%m-%d'),
            valid_native_observations=len(valid), missing_native_in_archive=missing_archive,
            missing_native_within_valid_span=int(native.loc[valid.index[0]:valid.index[-1]].isna().sum()),
            first_numeric_month=str(ms.index[0]), last_numeric_month=str(ms.index[-1]),
            available_months=len(ms), missing_months_within_record=int(interior.isna().sum()),
            complete_calendar_years=int((annual_counts == 12).sum())))
    inventory = pd.DataFrame(rows)
    assert len(inventory) == 12
    return inventory


def plot(inventory):
    plotted = inventory[inventory['index'] != 'OLREN'].sort_values('first_numeric_month', kind='stable')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'savefig.dpi': 300})
    fig, ax = plt.subplots(figsize=(7.5, 7.4))
    fig.subplots_adjust(left=.315, right=.765, top=.84, bottom=.10)
    fig.suptitle('Climate-index record coverage', y=.965, fontsize=14, va='center')
    fig.legend(handles=[Patch(facecolor='#2166AC', label='Available data'),
        Patch(facecolor='#ECEFF2', label='Outside record period')],
        loc='upper center', bbox_to_anchor=(.52,.935), ncol=2, frameon=False, fontsize=9)
    for y, row in enumerate(plotted.itertuples()):
        start, end = pd.Period(row.first_numeric_month), pd.Period(row.last_numeric_month)
        left, right = start.year+(start.month-1)/12, end.year+end.month/12
        assert row.available_months == len(pd.period_range(start,end,freq='M'))
        ax.barh(y,180,left=1850,height=.59,color='#ECEFF2',zorder=1)
        ax.barh(y,right-left,left=left,height=.59,color='#2166AC',zorder=3)
        label=f'{start.to_timestamp():%b %Y}\nto {end.to_timestamp():%b %Y}'
        ax.text(1.045,y,label,transform=ax.get_yaxis_transform(),va='center',fontsize=9.5,linespacing=1.15)
    labels=[textwrap.fill(s,25,break_long_words=False,break_on_hyphens=False) for s in plotted.full_name]
    ax.set(xlim=(1850,2030),ylim=(10.65,-.65),yticks=np.arange(11),yticklabels=labels,
        xticks=[1850,1900,1950,2000,2026],xlabel='Year')
    ax.tick_params(axis='y',length=0,pad=9,labelsize=10)
    ax.tick_params(axis='x',labelsize=9)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#888888')
    ax.set_axisbelow(True)
    ax.grid(axis='x',color='#D9D9D9',lw=.6)
    fig.text(.315,.865,'Available record',fontsize=9.5,weight='bold')
    fig.text(.785,.865,'Start and end',fontsize=9.5,weight='bold')
    for ext in ['png','pdf']:
        fig.savefig(FIGURES /f'figure_s01_climate_index_data_availability.{ext}',facecolor='white')
    plt.close(fig)
