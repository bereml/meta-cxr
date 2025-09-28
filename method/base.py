""" base.py """

import warnings
from argparse import Namespace

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn

from torchmetrics.functional.classification import binary_auroc, multilabel_auroc


def auroc(y_true, y_prob, average='micro'):
    n_classes = y_true.shape[1]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if n_classes == 1:
            return binary_auroc(y_prob.view(-1), y_true.view(-1))
        else:
            return multilabel_auroc(y_prob, y_true, n_classes, average)


@torch.inference_mode(True)
def compute_metrics(y_true, y_prob, seen, unseen, per_class):
    n_seen, n_unseen = len(seen), len(unseen)
    y_true = y_true.int()
    if n_seen and n_unseen:
        y_true_seen = y_true[:, :n_seen]
        y_prob_seen = y_prob[:, :n_seen]
        y_true_unseen = y_true[:, n_seen:]
        y_prob_unseen = y_prob[:, n_seen:]
        auroc_seen = auroc(y_true_seen, y_prob_seen).item() * 100
        auroc_unseen = auroc(y_true_unseen, y_prob_unseen).item() * 100
        auroc_hm = ((2 * auroc_seen * auroc_unseen) /
                    (auroc_seen + auroc_unseen))
    elif n_seen:
        y_true_seen = y_true[:, :n_seen]
        y_prob_seen = y_prob[:, :n_seen]
        auroc_seen = auroc(y_true_seen, y_prob_seen).item() * 100
        auroc_unseen = ''
        auroc_hm = auroc_seen
    else:
        y_true_unseen = y_true[:, n_seen:]
        y_prob_unseen = y_prob[:, n_seen:]
        auroc_seen = ''
        auroc_unseen = auroc(y_true_unseen, y_prob_unseen).item() * 100
        auroc_hm = auroc_unseen
    metrics = {
        'seen': auroc_seen,
        'unseen': auroc_unseen,
        'hm': auroc_hm
    }
    if per_class:
        auroc_classes = auroc(y_true, y_prob, 'none').cpu().numpy() * 100
        metrics.update(zip(seen + unseen, auroc_classes))
    return metrics


class FewShotMethod(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.loss_fn = self.build_loss()
        self.episodes_metrics = []
        float_type_dict = {16: torch.bfloat16, 32: torch.float32}
        self.float_type = float_type_dict[
            hparams['precision'] if isinstance(hparams, dict)
            else hparams.precision
        ]

    def convert_hparams(self, hparams):
        if isinstance(hparams, dict):
            return Namespace(**hparams)
        return hparams

    def save_hparams(self, hparams, net):
        cfg = net.backbone.pretrained_cfg
        hparams.norm = {'mean': list(cfg['mean']), 'std': list(cfg['std'])}
        self.save_hyperparameters(hparams)

    def build_loss(self):
        return nn.BCEWithLogitsLoss()

    def configure_optimizers(self):
        return None

    def on_test_epoch_end(self):
        episodes_dfs = []
        for episode_metrics in self.episodes_metrics:
            metrics = {'seed': self.hparams.seed}
            metrics.update(episode_metrics)
            episode_df = pd.Series(metrics).to_frame().T
            episodes_dfs.append(episode_df)
        df = pd.concat(episodes_dfs, axis=0, ignore_index=True)
        df['seed'] = df['seed'].astype(int)
        self.test_df = df

    def log_metrics(self, meta_set, metrics):
        metrics = {f'{k}/{meta_set}': v
                   for k, v, in metrics.items()
                   if isinstance(v, float)}
        metrics[f'loss/{meta_set}'] *= 100
        self.log_dict(metrics, on_epoch=True,
                      on_step=self.hparams['log_on_step'])

    def compute_metrics_and_log(self, meta_set, y_true, y_prob,
                                seen, unseen, loss):
        metrics = compute_metrics(y_true, y_prob, seen, unseen, False)
        metrics['loss'] = loss.item()
        self.log_metrics(meta_set, metrics)

    def compute_full_metrics(self, y_true_tst, y_prob_tst, seen, unseen):
        return compute_metrics(y_true_tst, y_prob_tst, seen, unseen, True)

    def add_episode_metrics(self, metrics):
        self.episodes_metrics.append(metrics)

    def split(self, episode):
        n_trn = episode['n_trn']
        # (n, 3, h, w)
        x = episode['x']
        # (n, c)
        y_true = episode['y']
        # split episode into trn/tst
        x_trn, y_true_trn = x[:n_trn], y_true[:n_trn]
        x_tst, y_true_tst = x[n_trn:], y_true[n_trn:]
        return x_trn, y_true_trn, x_tst, y_true_tst

    def advance_global_step(self):
        self.trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.increment_completed()


class Registry(dict):

    def register(self, name):
        def decorator_register(obj):
            self[name] = obj
            return obj
        return decorator_register


METHODS: dict[str, type[FewShotMethod]] = Registry()
