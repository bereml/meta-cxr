"""
Experiments for Publication
"""

import os
from itertools import product
from os.path import join

from tqdm import tqdm

from utils import adapt, aggregate_exp_df, load_config, pretrain_adapt


SEEDS = [0]
CHECKPOINT_NAME = 'mobilenetv3-small-075_i1k+batchbased.pth'
RESULTS_DIR = 'results'

DEBUG_HPARAMS_BB = {
    'batchbased_train_batches': 1,
    'mval_episodes': 1,
    'mtst_episodes': 1,
    'max_epochs': 1,
}

DEBUG_HPARAMS_PN = {
    'mtrn_episodes': 1,
    'mval_episodes': 1,
    'mtst_episodes': 1,
    'max_epochs': 1,
}

def debug_hparams(method: str = 'batchbased') -> dict[str, int]:
    return DEBUG_HPARAMS_BB if method == 'batchbased' else DEBUG_HPARAMS_PN


###############################################################################


def paper_arch(
        seeds=SEEDS,
        results_dir=RESULTS_DIR,
        debug=False):
    exp = 'arch'
    mtrn_batch_size = 32
    cfgs = list(product(
        # arch
        [
            # efficient
            'mobilenetv3-small-075',
            'mobilenetv3-large-100',
            'mobilevitv2-100',
            'convnext-atto',
            # large
            'densenet121',
            'densenet161',
            'convnext-tiny',
            'mobilevitv2-200',
        ],
        seeds,
    ))
    for cfg in tqdm(cfgs, desc=f'EXP {exp}', ncols=75):
        backbone, seed = cfg
        run = '_'.join([
            backbone,
        ])
        hparams = {}
        if debug:
            hparams.update(debug_hparams())
            results_dir = 'rdev'
        pretrain_adapt(
            results_dir=results_dir,
            exp=exp,
            run=run,
            net_backbone=backbone,
            mtrn_batch_size=mtrn_batch_size,
            seed=seed,
            **hparams
        )
        aggregate_exp_df(join(results_dir, exp))


def paper_base(
        seeds=SEEDS,
        results_dir=RESULTS_DIR,
        debug=False):
    exp = 'base'
    net_weights = 'i1k'
    cfgs = list(product(
        # net_backbone
        ['mobilenetv3-small-075', 'mobilenetv3-large-100'],
        # method
        ['batchbased', 'protonet'],
        seeds,
    ))
    for cfg in tqdm(cfgs, desc=f'EXP {exp}', ncols=75):
        net_backbone, method, seed = cfg
        run = '_'.join([
            net_backbone,
            net_weights,
            method,
        ])
        checkpoint_name = f'{net_backbone}_{net_weights}+{method}.pth'

        hparams = {}
        if debug:
            hparams.update(debug_hparams(method))
            results_dir = 'rdev'

        pretrain_adapt(
            results_dir=results_dir,
            exp=exp,
            run=run,
            net_backbone=net_backbone,
            net_weights=net_weights,
            method=method,
            seed=seed,
            checkpoint_name=checkpoint_name,
            **hparams
        )
        aggregate_exp_df(join(results_dir, exp))


def paper_pretraining(
        seeds=SEEDS,
        results_dir=RESULTS_DIR,
        debug=False):
    exp = 'pretraining'
    net_backbone = 'mobilenetv3-large-100'
    cfgs = list(product(
        #   [net_weights, method]
        [
            ['i1k',       'batchbased'],
            ['i1k',       'protonet'],
            ['i21k',      'batchbased'],
            ['i21k',      'protonet'],
            ['random',    'batchbased'],
            ['random',    'protonet'],
        ],
        seeds,
    ))
    for cfg in tqdm(cfgs, desc=f'EXP {exp}', ncols=75):
        (net_weights, method), seed = cfg
        run = '_'.join([
            net_weights,
            method,
        ])

        hparams = {}
        if debug:
            hparams.update(debug_hparams(method))
            results_dir = 'rdev'

        pretrain_adapt(
            results_dir=results_dir,
            exp=exp,
            run=run,
            net_backbone=net_backbone,
            net_weights=net_weights,
            method=method,
            seed=seed,
            **hparams
        )
        aggregate_exp_df(join(results_dir, exp))


def paper_resolution(
        seeds=SEEDS,
        results_dir=RESULTS_DIR,
        debug=False):
    exp = 'resolution'
    mtrn_batch_size = 32
    cfgs = list(product(
        # image_size, net_backbone
        [
            [ 224, 'mobilenetv3-small-075'],
            [ 384, 'mobilenetv3-small-075'],
            [ 512, 'mobilenetv3-small-075'],
            [ 768, 'mobilenetv3-small-075'],
            [1024, 'mobilenetv3-small-075'],
            [ 224, 'convnext-tiny'],
            [ 384, 'convnext-tiny'],
            [ 512, 'convnext-tiny'],
            [ 768, 'convnext-tiny'],
            [ 224, 'densenet121'],
            [ 384, 'densenet121'],
            [ 512, 'densenet121'],
        ],
        seeds,
    ))
    for cfg in tqdm(cfgs, desc=f'EXP {exp}', ncols=75):
        (image_size, net_backbone), seed = cfg
        run = '_'.join([
            net_backbone,
            f'{image_size:04d}',
        ])
        hparams = {}
        if debug:
            hparams.update(debug_hparams())
            results_dir = 'rdev'
        pretrain_adapt(
            results_dir=results_dir,
            exp=exp,
            run=run,
            image_size=image_size,
            net_backbone=net_backbone,
            mtrn_batch_size=mtrn_batch_size,
            seed=seed,
            **hparams
        )
        aggregate_exp_df(join(results_dir, exp))


def paper_task_complexity_bb(
        seeds=SEEDS,
        results_dir=RESULTS_DIR,
        checkpoint_name='mobilenetv3-small-075_i1k+batchbased.pth',
        debug=False):
    exp = 'task_complexity'
    method = 'batchbased'
    cfgs = list(product(
        # mtst_n_way, mtst_n_unseen
        [
            [3, 1],
            [3, 2],
            [3, 3],
            [4, 1],
            [4, 2],
            [4, 3],
            [4, 4],
            [5, 1],
            [5, 2],
            [5, 3],
            [5, 4],
            [5, 5],
        ],
        # mtst_trn_k_shot
        [1, 5, 15, 30],
        seeds,
    ))
    for cfg in tqdm(cfgs, desc=f'EXP {exp}', ncols=75):
        (mtst_n_way, mtst_n_unseen), mtst_trn_k_shot, seed = cfg
        run = '_'.join([
            method,
            f'nway-{mtst_n_way}',
            f'unseen-{mtst_n_unseen}',
            f'kshot-{mtst_trn_k_shot:02d}',
        ])
        hparams = {}
        if debug:
            hparams.update(debug_hparams(method))
            results_dir = 'rdev'
        adapt(
            results_dir=results_dir,
            exp=exp,
            run=run,
            mtst_n_way=mtst_n_way,
            mtst_n_unseen=mtst_n_unseen,
            mtst_trn_k_shot=mtst_trn_k_shot,
            method=method,
            seed=seed,
            checkpoint_name=checkpoint_name,
            **hparams
        )
        aggregate_exp_df(join(results_dir, exp))


def paper_task_complexity_pn(
        seeds=SEEDS,
        results_dir=RESULTS_DIR,
        checkpoint_name='mobilenetv3-small-075_i1k+protonet.pth',
        debug=False):
    exp = 'task_complexity'
    method = 'protonet'
    cfgs = list(product(
        # mtst_n_way, mtst_n_unseen
        [
            [3, 1],
            [3, 2],
            [3, 3],
            [4, 1],
            [4, 2],
            [4, 3],
            [4, 4],
            [5, 1],
            [5, 2],
            [5, 3],
            [5, 4],
            [5, 5],
        ],
        # mtst_trn_k_shot
        [1, 5, 15, 30],
        seeds,
    ))
    for cfg in tqdm(cfgs, desc=f'EXP {exp}', ncols=75):
        (mtst_n_way, mtst_n_unseen), mtst_trn_k_shot, seed = cfg
        run = '_'.join([
            method,
            f'nway-{mtst_n_way}',
            f'unseen-{mtst_n_unseen}',
            f'kshot-{mtst_trn_k_shot:02d}',
        ])
        hparams = {}
        if debug:
            hparams.update(debug_hparams(method))
            results_dir = 'rdev'

        adapt(
            results_dir=results_dir,
            exp=exp,
            run=run,
            mtst_n_way=mtst_n_way,
            mtst_n_unseen=mtst_n_unseen,
            mtst_trn_k_shot=mtst_trn_k_shot,
            method=method,
            seed=seed,
            checkpoint_name=checkpoint_name,
            **hparams
        )
        aggregate_exp_df(join(results_dir, exp))


##################################################


def main(debug=False):
    if not load_config()['metachest_dir']:
        print("Set metachest_dir in config.toml, see README.md.")
        exit(1)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    paper_base(debug=debug)
    paper_pretraining(debug=debug)
    paper_task_complexity_bb(debug=debug)
    paper_task_complexity_pn(debug=debug)
    paper_arch(debug=debug)
    paper_resolution(debug=debug)


if __name__ == '__main__':
    import fire
    fire.Fire(main)
