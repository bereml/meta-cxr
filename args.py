import argparse

from method import METHODS
from utils import str2bool, timestamp


def method_add_args(method, parser):
    Method = METHODS.get(method, None)
    if Method is None:
        raise ValueError(f'unknown method {method}')
    Method.add_args(parser)


def parse_args():
    parser = argparse.ArgumentParser()
    # RUN
    parser.add_argument('--results_dir',
                        type=str, default='rdev',
                        help='parent results, directory')
    parser.add_argument('--exp',
                        type=str, default='runs',
                        help='parent experiment directory')
    parser.add_argument('--run',
                        type=str, default=timestamp(),
                        help='run directory')
    parser.add_argument('--seed',
                        type=int, default=0,
                        help='random seed')
    # DATA
    parser.add_argument('--data_distro',
                        type=str, default='complete',
                        help='data distribution')
    parser.add_argument('--mtrn_batch_size',
                            type=int, default=64,
                            help='meta-trn batch size')
    parser.add_argument('--mtrn_n_way',
                        type=int, default=3,
                        help='meta-training n-way')
    parser.add_argument('--mtrn_trn_k_shot',
                        type=int, default=30,
                        help='meta-training training k-shot')
    parser.add_argument('--mtrn_tst_k_shot',
                        type=int, default=30,
                        help='meta-training test k-shot')
    parser.add_argument('--mval_n_way',
                        type=int, default=3,
                        help='meta-validation n-way')
    parser.add_argument('--mval_n_unseen',
                        type=int, default=1,
                        help='meta-validation n-way unseen')
    parser.add_argument('--mval_trn_k_shot',
                        type=int, default=30,
                        help='meta-validation training k-shot')
    parser.add_argument('--mval_tst_k_shot',
                        type=int, default=30,
                        help='Meta-validation test k-shot')
    parser.add_argument('--mtst_n_way',
                        type=int, default=3,
                        help='meta-test n-way')
    parser.add_argument('--mtst_n_unseen',
                        type=int, default=1,
                        help='meta-test n-way unseen')
    parser.add_argument('--mtst_trn_k_shot',
                        type=int, default=30,
                        help='meta-test training k-shot')
    parser.add_argument('--mtst_tst_k_shot',
                        type=int, default=30,
                        help='meta-tst test k-shot')
    parser.add_argument('--image_size',
                        type=int, default=384,
                        help='image size')
    parser.add_argument('--data_aug',
                        type=str2bool, default=False,
                        help='enable data augmentation')
    parser.add_argument('--num_workers',
                        type=int, default=8,
                        help='dataloaders number of workers')
    # NETWORK
    parser.add_argument('--net_backbone',
                        type=str, default='mobilenetv3-small-075',
                        help='backbone architecture')
    parser.add_argument('--net_weights',
                        type=str, default='i1k',
                        help='backbone pretrained weights')
    # METHOD
    parser.add_argument('--method',
                        type=str, default='batchbased',
                        help='learning method')
    # TRAINING
    parser.add_argument('--mtrn_episodes',
                        type=int, default=1_000,
                        help='number of meta-training episodes')
    parser.add_argument('--mval_episodes',
                        type=int, default=100,
                        help='number of meta-validation episodes')
    parser.add_argument('--mtst_episodes',
                        type=int, default=10_000,
                        help='number of meta-test episodes')
    parser.add_argument('--max_epochs',
                        type=int, default=100,
                        help='maximum number of epochs')
    parser.add_argument('--stop_metric',
                        type=str, default='hm',
                        choices=['hm', 'loss'],
                        help='early stopping metric')
    parser.add_argument('--stop_patience',
                        type=int, default=10,
                        help='early stopping patience')
    parser.add_argument('--pretrain_adapt',
                        type=str2bool, default=True, nargs='?', const=False,
                        help='pretrain and adapt the model')
    parser.add_argument('--checkpoint_name',
                        type=str, default=None,
                        help='checkpoint name')
    parser.add_argument('--checkpoints_dir',
                        type=str, default='checkpoints',
                        help='checkpoints directory')
    parser.add_argument('--accelerator',
                        type=str, default='auto',
                        choices=['auto', 'cpu', 'gpu'],
                        help='Accelerator')
    parser.add_argument('--devices', type=int,
                        default=1,
                        help='Devices')
    parser.add_argument('--benchmark',
                        type=str2bool, default=False, nargs='?', const=False,
                        help="cudnn benchmark")
    parser.add_argument('--deterministic',
                        type=str, default='warn',
                        choices=['warn', 'true', 'false'],
                        help='enforce deterministic algos')
    parser.add_argument('--precision',
                        type=int, default=16,
                        choices=[16, 32],
                        help='float precision')
    # DEBUG
    parser.add_argument('--log_on_step',
                        type=str2bool, default=False, nargs='?', const=False,
                        help='lightning log on step metrics')
    parser.add_argument('--debug',
                        type=str2bool, default=False, nargs='?', const=False,
                        help='debug mode')
    # CSVs
    parser.add_argument('--run_mtst_csv',
                        type=str, default='run_mtst.csv',
                        help='run csv results name')
    parser.add_argument('--seeds_mtst_csv',
                        type=str, default='seeds_mtst.csv',
                        help='seeds csv results name')
    parser.add_argument('--episodes_mtst_csv',
                        type=str, default='episodes_mtst.csv',
                        help='episodes csv results name')

    # method specific args
    parsed, _ = parser.parse_known_args()
    method_add_args(parsed.method, parser)
    # parse complete args
    parser = parser.parse_args()
    return parser
