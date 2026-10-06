"""Rebuild any numbered figure or table from the archived study workbook.

Run --help for commands. Source data and manuscripts are never overwritten.
"""
import sys
sys.dont_write_bytecode = True
import argparse
from pathlib import Path
from hashlib import sha256
import importlib
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cbook import boxplot_stats
from output_catalogue import FIGURES
from table_definitions import TABLES
from study_results import StudyResults
from table_results import rows_for

ROOT = Path(__file__).resolve().parents[1]


def draw(key, study, destination):
    modules = {
        '01':'thermodynamics', '02':'circulation_figures', '03':'sst_figures', '04':'sst_figures',
        '05':'index_dependence', '06':'dependence_figures', '07':'dependence_figures', '08':'el_nino',
        's01':'inventory', 's02':'winter_aggregation', 's03':'winter_diagnostics', 's04':'winter_diagnostics',
        's05':'circulation_figures', 's06':'circulation_figures', 's07':'sst_figures',
        's08':'dependence_figures', 's09':'dependence_figures', 's10':'dependence_figures',
        's11':'dependence_figures', 's12':'el_nino_figures'}
    number = key.split('_')[1]
    module = importlib.import_module(modules[number])
    module.FIGURES = destination
    if number=='01':
        module.HERE = destination
    # Reset between outputs: rendering must not depend on the preceding figure.
    with plt.rc_context(matplotlib.rcParamsDefault):
        plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'savefig.dpi':300})
        if number=='01': module.plot()
        elif number=='02': module.records(study.winter[0],study.trends)
        elif number=='03': module.trend_comparison(study.trends)
        elif number=='04': module.contrasts_and_variability(study.trends)
        elif number=='05': module.plot(*study.dependence[1:])
        elif number=='06': module.component_variance(study.dependence[0])
        elif number=='07': module.conditional_association(study.winter[2],study.dependence[0])
        elif number=='08': module.plot(study.events[1])
        elif number=='s01': module.plot(study.inventory)
        elif number=='s02': module.plot(study.winter[2],study.winter[3])
        elif number in ['s03','s04']:
            plt.rcParams.update({'axes.labelcolor':'#262626','text.color':'#262626'})
            if number=='s03':
                module.distributions(study.winter[1],[boxplot_stats(study.winter[1][k],whis=1.5)[0] for k in module.KEYS])
            else:
                df=study.frame('winter_index_autocorrelations')
                module.autocorrelations({k:df.loc[df['index']==k,'autocorrelation'].to_numpy() for k in module.KEYS})
        elif number=='s05': module.period_comparison(study.winter[0],study.trends)
        elif number=='s06': module.sensitivity(study.trends)
        elif number=='s07': module.winter_records(study.winter[0],study.trends)
        elif number=='s08': module.variance_inflation(study.dependence[0])
        elif number=='s09': module.strongest_pairs(study.winter[1])
        elif number=='s10': module.detrending(study.dependence[0],study.dependence[1])
        elif number=='s11': module.omission(study.dependence[0])
        elif number=='s12': module.plot_sensitivity(*study.events)


def normalize(value, kind):
    value = value.lower().removeprefix(kind+'_')
    return kind+'_'+('s'+value[1:].zfill(2) if value.startswith('s') else value.zfill(2))


def main(target=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    choice = parser.add_mutually_exclusive_group()
    if target is None:
        choice.add_argument('--figure', metavar='NUMBER', help='Main number or supplementary number, e.g. 5 or S10')
        choice.add_argument('--table', metavar='NUMBER', help='Main number or supplementary number, e.g. 3 or S20')
        choice.add_argument('--all', action='store_true', help='Rebuild all 20 figures and 28 tables')
    choice.add_argument('--list', action='store_true', help='List numbered outputs and their scripts')
    parser.add_argument('--output-dir', type=Path, help='Write to a separate folder containing Figures and Tables')
    parser.add_argument('--check', action='store_true', help='Compare all recalculated datasets with the saved CSV archive')
    parser.add_argument('--export-data', action='store_true', help='Export the recalculated CSVs to --output-dir/Data (requires a separate output folder)')
    args = parser.parse_args()
    catalogue = {**FIGURES, **{k:v['stem'] for k,v in TABLES.items()}}
    if args.list:
        for key,stem in catalogue.items():
            folder='Figures' if key.startswith('figure') else 'Tables'
            print(f'{key:10s} {folder}/generate_{stem}.py')
        return
    targets = [target] if target else list(catalogue) if args.all else (
        [normalize(args.figure,'figure')] if args.figure else [normalize(args.table,'table')] if args.table else [])
    if not targets and not (args.check or args.export_data):
        parser.error('Choose --figure, --table, --all, --check or --list.')
    if any(key not in catalogue for key in targets):
        parser.error('Unknown output number; use --list.')
    output = args.output_dir.resolve() if args.output_dir else ROOT
    if args.export_data and (not args.output_dir or output==ROOT):
        parser.error('--export-data requires a separate --output-dir; archived Data files are read-only.')
    if output==ROOT/'Data' or ROOT/'Data' in output.parents:
        parser.error('Do not use the source Data directory as an output location.')
    sources={p:sha256(p.read_bytes()).hexdigest() for p in (ROOT/'Data').iterdir() if p.is_file()}
    study=StudyResults()
    if args.check:
        study.check_archive()
    if args.export_data:
        data=output/'Data'; data.mkdir(parents=True,exist_ok=True)
        for name,frame in study.all_frames().items():
            frame.to_csv(data/name,index=False,encoding='utf-8-sig',float_format='%.17g')
    for key in targets:
        folder=output/('Figures' if key.startswith('figure') else 'Tables')
        folder.mkdir(parents=True,exist_ok=True)
        if key.startswith('figure'):
            draw(key,study,folder)
        else:
            from table_writer import write_table
            write_table(key,rows_for(key,study),folder)
        print(f'Saved {key}: {folder/catalogue[key]}')
    if any(sha256(p.read_bytes()).hexdigest()!=digest for p,digest in sources.items()):
        raise AssertionError('A source file changed during reproduction.')
    print('Source files unchanged.')


if __name__=='__main__':
    main()
