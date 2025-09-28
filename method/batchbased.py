""" batchbased.py """

from copy import deepcopy

import torch
import torch.optim as optim
from torch.amp import GradScaler

from network import create_net
from utils import str2list
from .base import FewShotMethod, METHODS


@METHODS.register('batchbased')
class BatchBased(FewShotMethod):

    def __init__(self, hparams):
        super().__init__(hparams)
        hparams = self.convert_hparams(hparams)
        self.net = create_net(
            backbone=hparams.net_backbone,
            weights=hparams.net_weights,
            checkpoints_dir=hparams.checkpoints_dir
        )
        self.save_hparams(hparams, self.net)
        self.automatic_optimization = False

    def configure_optimizers(self):
        opt = optim.AdamW(
            self.net.parameters(),
            lr=self.hparams.batchbased_trn_lr
        )
        return opt

    def on_train_start(self):
        super().on_train_start()
        self.scaler = GradScaler()

    def train_batch(self, batch):
        x, y_true = batch['x'], batch['y']
        with torch.autocast(self.device.type, self.float_type):
            y_lgts = self.net(x)
            loss = self.loss_fn(y_lgts, y_true)
        opt = self.optimizers()
        opt.zero_grad()
        loss = self.scaler.scale(loss)
        self.manual_backward(loss)
        self.scaler.step(opt)
        self.scaler.update()
        with torch.no_grad():
            y_prob = torch.sigmoid(y_lgts)
        return y_true, y_prob, loss


    def training_step(self, batch, _):
        if self.net.head_classes == 0:
            self.net.new_head('fc', len(batch['seen']) + len(batch['unseen']))
        y_true, y_prob, loss = self.train_batch(batch)
        if (self.hparams.batchbased_reset_head != 0 and
            (self.current_epoch + 1) % self.hparams.batchbased_reset_head == 0):
            self.net.reset_head()
        self.compute_metrics_and_log('mtrn', y_true, y_prob,
                                     batch['seen'], batch['unseen'], loss)


    def adapt_episode_inner(self, x, y_true, net, opt, steps, batch_size):
        scaler = GradScaler()
        for _ in range(steps):
            idx = torch.randperm(x.shape[0])[:batch_size]
            x_batch, y_true_batch = x[idx], y_true[idx]
            with torch.autocast(self.device.type, self.float_type):
                y_lgts_batch = net(x_batch)
                loss = self.loss_fn(y_lgts_batch, y_true_batch)
            opt.zero_grad()
            loss = scaler.scale(loss)
            # same results with loss.backward() or self.manual_backward(loss)
            loss.backward()
            scaler.step(opt)
            scaler.update()


    def adapt_episode(self, episode):
        # prepare data & net
        x_trn, y_true_trn, x_tst, y_true_tst = self.split(episode)
        n_examples, n_classes = y_true_trn.shape
        net = deepcopy(self.net)
        net.new_head('fc', n_classes)

        # adapt full net
        batch_size = int(n_examples *
            self.hparams.batchbased_mval_net_batch_pct)
        net.unfreeze_and_train_backbone()
        net.unfreeze_and_train_head()
        opt = optim.AdamW(net.parameters(),
                          lr=self.hparams.batchbased_mval_net_lr)
        self.adapt_episode_inner(
            x_trn, y_true_trn, net, opt,
            self.hparams.batchbased_mval_net_steps, batch_size)

        # adapt head only
        batch_size = int(n_examples *
            self.hparams.batchbased_mval_head_batch_pct)
        net.freeze_and_eval_backbone()
        opt = optim.AdamW(net.head.parameters(),
                          lr=self.hparams.batchbased_mval_head_lr)
        self.adapt_episode_inner(
            x_trn, y_true_trn, net, opt,
            self.hparams.batchbased_mval_head_steps, batch_size)

        # evaluation
        net.eval()
        with torch.no_grad():
            y_lgts_tst = net(x_tst)
            y_prob_tst = torch.sigmoid(y_lgts_tst)
            loss = self.loss_fn(y_lgts_tst, y_true_tst)

        return y_true_tst, y_prob_tst, loss

    @torch.enable_grad()
    def validation_step(self, episode, _):
        y_true_tst, y_prob_tst, loss = self.adapt_episode(episode)
        self.compute_metrics_and_log(
            'mval', y_true_tst, y_prob_tst,
            episode['seen'], episode['unseen'],
            loss
        )

    @torch.enable_grad()
    def test_step(self, episode, _):
        y_true_tst, y_prob_tst, _ = self.adapt_episode(episode)
        metrics = self.compute_full_metrics(
            y_true_tst, y_prob_tst, episode['seen'], episode['unseen'])
        self.add_episode_metrics(metrics)

    @staticmethod
    def add_args(parser):
        # mtrn
        parser.add_argument('--batchbased_trn_lr',
                            type=float, default=0.0001,
                            help='meta-trn lr')
        parser.add_argument('--batchbased_sch_milestones',
                            type=str2list, default=[1],
                            help='scheduler milestones')
        parser.add_argument('--batchbased_reset_head',
                            type=int, default=0,
                            help='reset head every given epochs')
        # mval
        parser.add_argument('--batchbased_mval_net_batch_pct',
                            type=float, default=1.00,
                            help='meta-val data batch percentage used for inner step')
        parser.add_argument('--batchbased_mval_net_lr',
                            type=float, default=0.005,
                            help='meta-val full net learning rate')
        parser.add_argument('--batchbased_mval_net_steps',
                            type=int, default=0,
                            help='meta-val full net learning steps')
        parser.add_argument('--batchbased_mval_head_batch_pct',
                            type=float, default=0.50,
                            help='data batch percentage used for inner step')
        parser.add_argument('--batchbased_mval_head_lr',
                            type=float, default=0.005,
                            help='meta-val head only learning rate')
        parser.add_argument('--batchbased_mval_head_steps',
                            type=int, default=100,
                            help='meta-val head only learning steps')
        # other
        parser.add_argument('--batchbased_train_batches',
                            type=int, default=0,
                            help='number of train batches, 0 means all')
