""" train.py """

import os
import shutil
import warnings
from os.path import isdir, join

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import seed_everything

from args import parse_args
from data import build_dl, build_mdl
from adapt import adapt
from method import METHODS
from utils import RunTimer, get_run_dir


def pretrain_adapt():
    print("==================================\n"
          "=========== PRETRAIN =============")

    torch.set_float32_matmul_precision('medium')

    hparams = parse_args()
    seed_everything(hparams.seed, workers=True)

    run_dir = get_run_dir(hparams)
    if isdir(run_dir):
        print(f"Run already exists {run_dir}")
        return

    Method = METHODS.get(hparams.method, None)
    if Method is None:
        raise ValueError(f"unknown method {hparams.method}")
    method = Method(hparams)

    limit_train_batches = None
    if hparams.method in {'batchbased'}:
        mtrn_dl = build_dl(
            'mtrn',
            hparams.mtrn_batch_size,
            hparams
        )
        if hparams.batchbased_train_batches:
            limit_train_batches = hparams.batchbased_train_batches
    else:
        mtrn_dl = build_mdl(
            'mtrn',
            hparams.mtrn_episodes,
            hparams.mtrn_n_way,
            0,
            hparams.mtrn_trn_k_shot,
            hparams.mtrn_tst_k_shot,
            hparams
        )

    mval_dl = build_mdl(
        'mval',
        hparams.mval_episodes,
        hparams.mval_n_way,
        hparams.mval_n_unseen,
        hparams.mval_trn_k_shot,
        hparams.mval_tst_k_shot,
        hparams
    )

    monitor_mode = "min" if hparams.stop_metric == "loss" else "max"
    stop_metric = f"{hparams.stop_metric}/mval"

    checkpoint_cb = ModelCheckpoint(monitor=stop_metric, mode=monitor_mode)
    early_cb = EarlyStopping(
        monitor=stop_metric, patience=hparams.stop_patience, mode=monitor_mode
    )
    logger = TensorBoardLogger(
        join(hparams.results_dir, hparams.exp),
        hparams.run,
        version=f"seed{hparams.seed}",
        default_hp_metric=False,
    )
    deterministic = {
        'warn': 'warn', 'true': True, 'false': False
    }[hparams.deterministic]

    trainer_args = {}
    if hparams.precision == 16 and method.automatic_optimization:
        trainer_args['precision'] = 'bf16-mixed'

    trainer = pl.Trainer(
        accelerator=hparams.accelerator,
        benchmark=hparams.benchmark,
        callbacks=[checkpoint_cb, early_cb],
        deterministic=deterministic,
        devices=hparams.devices,
        logger=logger,
        log_every_n_steps=1,
        limit_train_batches=limit_train_batches,
        max_epochs=hparams.max_epochs,
        num_sanity_val_steps=0,
        inference_mode=False,
        **trainer_args,
    )

    # Ignore warnings for Protonet
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trainer.fit(method, mtrn_dl, mval_dl)
    # trainer.fit(method, mtrn_dl, mval_dl)

    best_checkpoint_path = checkpoint_cb.best_model_path
    print(f"Best: {best_checkpoint_path}")

    if hparams.checkpoint_name:
        os.makedirs(hparams.checkpoints_dir, exist_ok=True)
        checkpoint_path = join(
            hparams.checkpoints_dir, hparams.checkpoint_name)
        method = Method.load_from_checkpoint(
            best_checkpoint_path, strict=False)
        torch.save(method.net.backbone.state_dict(), checkpoint_path)
        # TODO: remove if we are going to preserve state dict way
        # shutil.copyfile(
        #     best_checkpoint_path,
        #     checkpoint_path
        # )
        print(f'Checkpoint saved to {checkpoint_path}')

    if hparams.pretrain_adapt:
        adapt(True, hparams, best_checkpoint_path)


def main():
    with RunTimer():
        pretrain_adapt()


if __name__ == "__main__":
    import sys
    sys.exit(main())
