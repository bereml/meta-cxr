""" utils.py

Utilities.
"""

import argparse
import datetime
import glob
import subprocess
import time
import tomllib
from collections import defaultdict
from os.path import join, isdir, isfile

import numpy as np
import pandas as pd
from scipy.stats import bootstrap


LINE = '=' * 75
PLUS_MINUS = '±'

_config = None


def compute_mean_ci(a, rt='pair', c=1.96):
    mean = np.mean(a)
    std = np.std(a)
    ci = c * std / np.sqrt(len(a))
    if rt == 'pair':
        return mean, ci
    elif rt == 'str':
        return f'{mean:.2f}{PLUS_MINUS}{ci:.2f}'
    elif rt == 'int':
        return mean-ci, mean+ci
    else:
        raise ValueError(f'invalid return type rt={rt}')


def agregate_seeds_df(episodes_df):
    data = defaultdict(list)
    for seed, df in episodes_df.groupby('seed'):
        data['seed'].append(seed)
        for col_name, col_vals in df.iloc[:, 1:].T.iterrows():
            values = col_vals.dropna().values
            if len(values) > 0:
                mean_ci = compute_mean_ci(values, rt='str')
            else:
                mean_ci = np.nan
            data[col_name].append(mean_ci)
    seeds_df = pd.DataFrame.from_dict(data)
    return seeds_df


def agregate_run_df_aux(df):
    data = {}
    for col_name, col_vals in df.iloc[:, 1:].T.iterrows():
        values = col_vals.dropna().values
        if len(values) == 0:
            mean_ci = ''
        elif len(values) == 1:
            mean_ci = values[0]
        else:
            values = [float(v[:5]) for v in values]
            mean = np.mean(values)
            ci = bootstrap([values], np.mean).confidence_interval
            ci = abs(mean - max(ci.low, ci.high))
            mean_ci = f'{mean:.2f}{PLUS_MINUS}{ci:.2f}'
        data[col_name] = [mean_ci]
    run_df = pd.DataFrame.from_dict(data)
    return run_df


def aggregate_run_df(run_dir,
                     run_mtst_csv='run_mtst.csv',
                     seeds_mtst_csv='seeds_mtst.csv',
                     episodes_mtst_csv='episodes_mtst.csv'):
    path = join(run_dir, episodes_mtst_csv)
    if not isfile(path):
        return None

    episodes_df = pd.read_csv(path)
    seeds_df = agregate_seeds_df(episodes_df)
    seeds_df.to_csv(join(run_dir, seeds_mtst_csv), index=False)


    run_df = agregate_run_df_aux(seeds_df)
    run_df.to_csv(join(run_dir, run_mtst_csv), index=False)

    return run_df


def aggregate_exp_df(exp_dir,
                     exp_mtst_csv='exp_mtst.csv',
                     run_mtst_csv='run_mtst.csv',
                     seeds_mtst_csv='seeds_mtst.csv',
                     episodes_mtst_csv='episodes_mtst.csv',
                     exp_mtst_md='exp_mtst.md',
                     exp_mtst_tex='exp_mtst.tex'):
    pattern = join(exp_dir, '*')
    runs_dirs = sorted(glob.glob(pattern, recursive=False))
    runs_dirs = [run_dir for run_dir in runs_dirs if isdir(run_dir)]
    if runs_dirs:
        dfs = []
        for run_dir in runs_dirs:
            df = aggregate_run_df(run_dir, run_mtst_csv,
                                  seeds_mtst_csv, episodes_mtst_csv)
            if df is not None:
                run = run_dir.split('/')[-1]
                df.insert(0, 'run', run)
                dfs.append(df)
        exp_df = pd.concat(dfs)
        exp_df.to_csv(join(exp_dir, exp_mtst_csv), index=False)
        # save df main columns on md & tex
        exp_df_overview = exp_df.iloc[:, :4]
        with open(join(exp_dir, exp_mtst_md), 'w') as f:
            f.write(exp_df_overview.to_markdown(index=False) + '\n')
        with open(join(exp_dir, exp_mtst_tex), 'w') as f:
            format_pm = lambda s: s.replace('±', '$\\pm$') if '±' in s else s
            formatters = [format_pm] * exp_df_overview.shape[1]
            f.write(exp_df_overview.to_latex(index=False, formatters=formatters))


def get_run_dir(hparams):
    return join(hparams.results_dir, hparams.exp,
                hparams.run, f'seed{hparams.seed}')


def load_config():
    global _config
    if _config is None:
        _config = read_toml('config.toml')
    return _config


def read_toml(path):
    with open(path, 'rb') as f:
        return tomllib.load(f)


def run_cmd(cmd, verbose=True):
    """Runs a command in the shell."""
    if verbose:
        print('\n' + LINE + '\n' + cmd)
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as cpe:
        print(cpe)
    if verbose:
        print(LINE)


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in {'yes', 'true', 't', 'y', '1'}:
        return True
    elif v in {'no', 'false', 'f', 'n', '0'}:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2list(s):
    return [int(i) for i in s.split(',')] if s else []


def timestamp(fmt='%y%m%dT%H%M%S'):
    """Returns current timestamp."""
    return datetime.datetime.fromtimestamp(time.time()).strftime(fmt)


def pretrain_adapt(**kwargs):
    elems = ['python pretrain.py'] + [f'--{k} {v}' for k, v in kwargs.items()]
    cmd = ' '.join(elems)
    run_cmd(cmd)


def adapt(**kwargs):
    elems = ['python adapt.py'] + [f'--{k} {v}' for k, v in kwargs.items()]
    cmd = ' '.join(elems)
    run_cmd(cmd)


class RunTimer:

    def __enter__(self):
        self.start = datetime.datetime.now()

    def __exit__(self, type, value, traceback):
        start, end = self.start, datetime.datetime.now()
        elapsed = end - start
        start = start.strftime('%DT%T')
        end = end.strftime('%DT%T')
        print(f"Runtime: {start} - {end} elapsed {elapsed}")


if __name__ == '__main__':
    import fire
    fire.Fire()
