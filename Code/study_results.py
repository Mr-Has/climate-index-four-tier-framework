"""Calculate manuscript results from the archived workbook without changing inputs."""
from functools import cached_property
from pathlib import Path
from hashlib import sha256
import numpy as np
import pandas as pd
import winter_aggregation
import trend_analysis
import index_dependence
import inventory
import el_nino
import thermodynamics

ROOT = Path(__file__).resolve().parents[1]
WORKBOOK_SHA256 = 'f5223da118589c1723347173f4563b4f1966cd0ab945cf50b934900d13dfcda6'


class StudyResults:
    """Cache calculation families so a complete rebuild does each calculation once."""
    def __init__(self):
        source = ROOT/'Data/Climate_Indices_Data.xlsx'
        actual = sha256(source.read_bytes()).hexdigest()
        if actual != WORKBOOK_SHA256:
            raise ValueError('The workbook differs from the manuscript archive. Review the source version before rebuilding.')

    @cached_property
    def winter(self):
        return winter_aggregation.calculate()

    @cached_property
    def trends(self):
        return trend_analysis.calculate(self.winter[0])

    @cached_property
    def dependence(self):
        return index_dependence.calculate(self.winter[1])

    @cached_property
    def events(self):
        return el_nino.calculate(self.winter[0])

    @cached_property
    def inventory(self):
        return inventory.calculate()

    def frame(self, name):
        filename = name + '.csv'
        if name.startswith('el_nino_'):
            return self.events[0][filename]
        if filename in ('winter_trend_results.csv', 'winter_trend_period_sensitivity.csv',
                        'winter_period_mean_comparisons.csv', 'winter_period_difference_uncertainty.csv',
                        'atlantic_pacific_trend_contrasts.csv', 'nino34_winter_variability_uncertainty.csv'):
            return self.trends[filename]
        return self.dependence[0][filename]

    def all_frames(self):
        t = np.linspace(-30, 40, 701)
        return {**self.trends, **self.dependence[0], **self.events[0],
            'climate_index_inventory.csv': self.inventory,
            'winter_climate_indices_1983_2022.csv': self.winter[1].rename_axis('winter_year').reset_index(),
            'winter_climate_index_statistics_1983_2022.csv': self.winter[3],
            'figure_01_thermodynamic_basis.csv': pd.DataFrame({
                'temperature_degC': t, 'saturation_vapour_pressure_hPa': thermodynamics.saturation(t),
                'local_sensitivity_percent_per_K': thermodynamics.sensitivity(t)})}

    def check_archive(self):
        """Check every derived CSV, including labels, shapes and all numeric columns."""
        frames = self.all_frames()
        for name, actual in frames.items():
            expected = pd.read_csv(ROOT/'Data'/name, float_precision='round_trip', keep_default_na=False)
            if list(actual.columns) != list(expected.columns) or actual.shape != expected.shape:
                raise AssertionError(f'{name}: schema or shape differs')
            for key in actual:
                if pd.api.types.is_numeric_dtype(actual[key]):
                    np.testing.assert_allclose(actual[key].to_numpy(float),
                        pd.to_numeric(expected[key], errors='coerce').to_numpy(float),
                        rtol=1e-10, atol=1e-12, equal_nan=True, err_msg=f'{name}: {key}')
                else:
                    a = actual[key].fillna('').astype(str).tolist()
                    b = expected[key].fillna('').astype(str).tolist()
                    if a != b:
                        raise AssertionError(f'{name}: {key} labels differ')
        print(f'Checked {len(frames)} derived datasets against the saved archive.')
        return frames
