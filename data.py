import random
from os.path import join, isdir, isfile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision.io import ImageReadMode, read_image

from utils import load_config


IMAGE_SIZES = {224, 336, 384, 448, 512, 768, 1024}
TRN_IDX, TST_IDX = 0, 1
METADATA_COLS = ['dataset', 'name', 'age', 'sex', 'view']


def _filter_mset(mset, mclasses, df):
    """ Filter df for the given mset:

        | mset | definition     |
        |------|----------------|
        | mtrn | (mval ∪ mtst)' |
        | mval | mval ∩ mtst'   |
        | mtst | mval' ∩ mtst   |

        See mtl_complete.ipynb
    """
    mtrn_classes = mclasses['mtrn']
    mval_classes = mclasses['mval']
    mtst_classes = mclasses['mtst']
    mval_mask = df[mval_classes].any(axis=1)
    mtst_mask = df[mtst_classes].any(axis=1)

    if mset == 'mtrn':
        # (mval ∪ mtst)'
        mset_mask = ~(mval_mask | mtst_mask)
        mset_classes = mtrn_classes
    elif mset == 'mval':
        # mval ∩ mtst'
        mset_mask = mval_mask & ~mtst_mask
        mset_classes = mtrn_classes + mval_classes
    else:
        # mval' ∩ mtst
        mset_mask = ~mval_mask & mtst_mask
        mset_classes = mtrn_classes + mtst_classes

    df = df[mset_mask].copy()
    cols = METADATA_COLS + mset_classes
    df = df[cols]
    return df


def _load_data(config, mset, distro):
    """ Load seen and unseen classes, and df with samples."""

    metachest_dir = config['metachest_dir']
    df_path = join(metachest_dir, f'metachest.csv')
    if not isfile(df_path):
        raise ValueError(f"MetaChest CSV not found {df_path}")
    df = pd.read_csv(df_path)

    # aplies the distro filter
    if distro != 'complete':
        distro_path = join(metachest_dir, 'distro', f'{distro}.csv')
        if not isfile(distro_path):
            raise ValueError(f"Distro CSV not found {distro_path}")
        distro_df = pd.read_csv(distro_path)
        distro_mask = distro_df[mset].astype(bool)
        df = df[distro_mask]

    # filter exaples
    mclasses = {'mtrn': config['mtrn'],
                'mval': config['mval'],
                'mtst': config['mtst']}
    df = _filter_mset(mset, mclasses, df)

    seen, unseen = {
        #        seen              unseen
        'mtrn': ([],               mclasses['mtrn']),
        'mval': (mclasses['mtrn'], mclasses['mval']),
        'mtst': (mclasses['mtrn'], mclasses['mtst']),
    }[mset]

    # filter out empty classes
    seen = [clazz for clazz in seen if df[clazz].any()]
    unseen = [clazz for clazz in unseen if df[clazz].any()]

    df = df[['dataset', 'name'] + seen + unseen]
    df[seen + unseen] = df[seen + unseen].fillna(0).astype(int)

    return seen, unseen, df


class XRayDataset(Dataset):
    """XRay Dataset batch version."""

    def __init__(self, mset, tsfm, hparams):
        """
        Parameters
        ----------

        mset : {'mtrn'}
            Meta-dataset.
        tsfm : Callable
            Image transformation.
        hparams : SimpleNamespace
            data_distro : str
                Data distribution name.
            image_size : int
                Image size.
        """

        if mset not in {'mtrn'}:
            raise ValueError(f'Invalid mset={mset}')
        if hparams.image_size not in IMAGE_SIZES:
            raise ValueError(f'Invalid image_size={hparams.image_size}')

        config = load_config()
        metachest_dir = config['metachest_dir']
        images_dir = join(metachest_dir, f'images-{hparams.image_size}')
        if not isdir(images_dir):
            raise ValueError(f'Dir not found images_dir={images_dir}')

        seen, unseen, df = _load_data(config, mset, hparams.data_distro)
        df = self._add_nf_data(metachest_dir, df, hparams.data_distro)

        self.mset = mset
        self.df = df
        self.seen = seen
        self.unseen = unseen
        self.images_dir = images_dir
        self.tsfm = tsfm
        self.samples = df[['dataset', 'name']].values.tolist()
        self.classes = df.columns[2:].tolist()
        self.labels = df[self.classes].to_numpy()

    def _add_nf_data(self, metachest_dir, df, distro):
        """" Add no-finding metachest_nf to df."""

        nf_df_path = join(metachest_dir, f'metachest_nf.csv')
        nf_df = pd.read_csv(nf_df_path)

        # aplies distro filter
        if distro != 'complete':
            distro_path = join(metachest_dir, 'distro', f'{distro}_nf.csv')
            if not isfile(distro_path):
                raise ValueError(f"Distro CSV not found {distro_path}")
            nf_distro_df = pd.read_csv(distro_path)
            nf_distro_mask = nf_distro_df['mask'].astype(bool)
            nf_df = nf_df[nf_distro_mask]

        # keep mset rows
        mset_id = 0
        nf_df = nf_df[nf_df['mset'] == mset_id]
        # keep only dataset and name columns
        nf_df = nf_df[['dataset', 'name']]

        # add 0 labels to nf_df
        classes = df.columns[2:].tolist()
        nf_df_labels = pd.DataFrame(
            np.zeros([len(nf_df), len(classes)], dtype=int),
            index=nf_df.index,
            columns=classes
        )
        nf_df = pd.concat([nf_df, nf_df_labels], axis=1)

        # add nf_df to df
        total_df = pd.concat([df, nf_df])

        return total_df


    def __getitem__(self, i):
        dataset, name = self.samples[i]

        image_path = join(self.images_dir, dataset, f'{name}.jpg')
        image = read_image(image_path, ImageReadMode.RGB)
        x = self.tsfm(image)

        y = self.labels[i]
        y = torch.tensor(y, dtype=torch.float32)

        example = [self.seen, self.unseen, dataset, name, x, y]
        return example


    def __len__(self):
        return len(self.samples)


class XRayMetaDatatset(Dataset):
    """XRay Meta-Dataset episode version."""

    def __init__(self, mset, trn_tsfm, tst_tsfm, hparams):
        """
        Parameters
        ----------

        mset : {'mtrn', 'mval', 'mtst'}
            Meta-dataset to load.
        trn_tsfm : Callable
            Train transformation.
        tst_tsfm : Callable
            Test transformation.
        hparams : SimpleNamespace
            data_distro : str
                Data distribution name.
            image_size : int
                Image size.
        """

        if mset not in {'mtrn', 'mval', 'mtst'}:
            raise ValueError(f'invalid mset={mset}')
        if hparams.image_size not in IMAGE_SIZES:
            raise ValueError(f'invalid image_size={hparams.image_size}')

        config = load_config()
        metachest_dir = config['metachest_dir']
        images_dir = join(metachest_dir, f'images-{hparams.image_size}')
        if not isdir(images_dir):
            raise ValueError(f'invalid images_dir={images_dir}')

        seen, unseen, df = _load_data(config, mset, hparams.data_distro)
        nf_df = self._load_nf_data(metachest_dir, mset, hparams.data_distro)

        self.mset = mset
        self.df = df
        self.seen = seen
        self.unseen = unseen
        self.nf_df = nf_df
        self.images_dir = images_dir
        self.trn_tsfm = trn_tsfm
        self.tst_tsfm = tst_tsfm
        self.tsfm = [trn_tsfm, tst_tsfm]


    def _load_nf_data(self, metachest_dir, mset, distro):
        """" Load no-finding metachest_nf df."""

        nf_df_path = join(metachest_dir, f'metachest_nf.csv')
        nf_df = pd.read_csv(nf_df_path)

        # aplies distro filter
        if distro != 'complete':
            distro_path = join(metachest_dir, 'distro', f'{distro}_nf.csv')
            if not isfile(distro_path):
                raise ValueError(f"Distro CSV not found {distro_path}")
            nf_distro_df = pd.read_csv(distro_path)
            nf_distro_mask = nf_distro_df['mask'].astype(bool)
            nf_df = nf_df[nf_distro_mask]

        # keep mset rows
        mset_id = {'mtrn': 0, 'mval': 1, 'mtst': 2}[mset]
        nf_df = nf_df[nf_df['mset'] == mset_id]
        # keep only dataset and name columns
        nf_df = nf_df[['dataset', 'name']]

        return nf_df


    def __getitem__(self, example):
        """Returns the example.

        Parameters
        ----------
        example : [int, str, str, [str], [str], [int]]
            Example with [subset, dataset, name, seen, unseen, labels]

        Returns
        -------
        [int, [str], [str], str, torch.tensor, torch.tensor]
            A list with [subset, seen, unseen, dataset, name, x, y].
        """
        subset, dataset, name, seen, unseen, labels = example

        image_path = join(self.images_dir, dataset, f'{name}.jpg')
        image = read_image(image_path, ImageReadMode.RGB)
        x = self.tsfm[subset](image)

        y = torch.tensor(labels, dtype=torch.float32)

        example = [subset, seen, unseen, dataset, name, x, y]
        return example

    def __len__(self):
        return len(self.df)


def sample_at_least_k_shot(df, nf_df, k_shot):
    """Samples an episode with at least `k` examples per class.

        Parameters
        ----------
        df : pd.DataFrame
            Array of labels with first column as index.
        k_shot : int
            Numbers of k-shot examples for the episode.

        Returns
        -------
        pd.DataFrame
            Episode dataframe.
        """

    # select examples
    episode_df = pd.DataFrame(columns=df.columns)
    classes = df.columns[2:].values
    for clazz in classes:
        # select missing examples for the class
        k_miss = k_shot - episode_df[clazz].sum()
        if k_miss > 0:
            # select the k missing examples
            class_df = df[df[clazz] == 1].iloc[:k_miss]
            # append them to the episode
            episode_df = pd.concat([episode_df, class_df])
            # remove them from the source
            df.drop(class_df.index, inplace=True)

    # sampĺe missing nf examples, at least 1 (due to unseen first sort)
    k_miss = max((len(classes) * k_shot) - episode_df.shape[0], 1)
    miss_nf_df = nf_df.sample(k_miss)
    nf_df_labels = pd.DataFrame(
        np.zeros([k_miss, len(classes)], dtype=int),
        index=miss_nf_df.index,
        columns=classes
    )
    miss_nf_df = pd.concat([miss_nf_df, nf_df_labels], axis=1)

    # append nf to the episode
    episode_df = pd.concat([episode_df, miss_nf_df])

    return episode_df


class EpisodeSampler(Sampler):
    """Multi-label episode sampler."""

    def __init__(self, dataset, n_episodes, n_way, n_unseen, trn_k_shot, tst_k_shot):
        """
        Parameters
        ----------
        dataset : XRayMetaDatatset
            The dataset.
            labels : np.ndarray
                Dataset labels.
        n_episodes : int
            Number of episodes to generate.
        n_way : int
            Number of classes for episode.
        trn_k_shot : int
            Minimal number of examples per classes for episode in training.
        tst_k_shot : int
            Minimal number of examples per classes for episode in testing.
        """
        if n_unseen > n_way:
            raise ValueError(
                f'Metaset {dataset.mset}: n_unseen={n_unseen} > n_way={n_way}')
        self.df = dataset.df
        self.seen = list(dataset.seen)
        self.unseen = list(dataset.unseen)
        self.n_episodes = n_episodes
        self.n_way = n_way
        self.trn_k_shot = trn_k_shot
        self.tst_k_shot = tst_k_shot
        if self.seen:
            self.n_unseen = min(n_unseen, len(self.unseen))
            self.n_seen = n_way - self.n_unseen
        else:
            self.n_unseen = n_way
            self.n_seen = 0

        self.nf_df = dataset.nf_df


    def generate_episode(self):
        """Generates an episode.

        Returns
        -------
        [int, str, str, [str], [str], [int]]
            A list of [subset, dataset, name, seen, unseen, labels] for each example.
        """

        # repeat sampling until there are enough examples per class
        while True:
            # sample classes
            random.shuffle(self.seen)
            random.shuffle(self.unseen)
            seen = self.seen[:self.n_seen]
            unseen = self.unseen[:self.n_unseen]

            # filter out examples with excluded classes
            excluded_classes = self.seen[self.n_seen:] + self.unseen[self.n_unseen:]
            excluded_mask = self.df[excluded_classes].any(axis=1)
            df = self.df[~excluded_mask]
            df = df[['dataset', 'name'] + seen + unseen].copy()

            # break if there are enough examples per class
            if (df[seen + unseen].sum(axis=0) > self.trn_k_shot).all():
                break

        # sort classes by ascending frequency
        # unseen go first to enable unseen-only examples
        # to be select on sample_at_least_k_shot()
        sorted_seen = list(df[seen].sum(axis=0).sort_values(ascending=True).index)
        sorted_unseen = list(df[unseen].sum(axis=0).sort_values(ascending=True).index)
        df = df[['dataset', 'name'] + sorted_unseen + sorted_seen]

        # shuffle dataset
        df = df.sample(df.shape[0])

        # sample episode subsets
        trn_df = sample_at_least_k_shot(df, self.nf_df, self.trn_k_shot)
        trn_df['set'] = trn_df.shape[0] * [TRN_IDX]
        tst_df = sample_at_least_k_shot(df, self.nf_df, self.tst_k_shot)
        tst_df['set'] = tst_df.shape[0] * [TST_IDX]
        episode_df = pd.concat([trn_df, tst_df])

        # restore original order of seen and unseen
        classes = seen + unseen

        # assamble episode
        episode_df = episode_df[['set', 'dataset', 'name'] + classes]
        episode = [
            [
                example['set'],
                example['dataset'],
                example['name'],
                seen,
                unseen,
                [example[c] for c in classes],
            ]
            for example in episode_df.to_dict('records')
        ]

        return episode

    def __iter__(self):
        """Yields a new episode.

        Yields
        -------
        [[int, [int], int]]
            A list of subset, classes and index for each example.
        """
        for _ in range(self.n_episodes):
            episode = self.generate_episode()
            yield episode

    def __len__(self):
        return self.n_episodes


def build_tsfm(data_aug, hparams, debug):
    tsfm = []

    if data_aug:
        tsfm.extend([
            T.RandomAffine(degrees=15,
                           translate=(0.1, 0.1),
                           scale=(0.9, 1.1))
        ])
    if not debug:
        mean, std = hparams.norm['mean'], hparams.norm['std']
        tsfm.extend([
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean, std),
        ])
    tsfm = nn.Sequential(*tsfm)
    return tsfm


def collate_batch(batch):
    datasets, names, xs, ys = [], [], [], []
    for seen, unseen, dataset, name, x, y in batch:
        datasets.append(dataset)
        names.append(name)
        xs.append(x)
        ys.append(y)
    x = torch.stack(xs)
    y = torch.stack(ys)
    episode = {
        'seen': seen,
        'unseen': unseen,
        'dataset': datasets,
        'name': names,
        'x': x,
        'y': y
    }
    return episode


def collate_episode(episode):
    # assuming TRN_IDX, TST_IDX = 0, 1
    size = [0, 0]
    datasets, names, xs, ys = [], [], [], []
    for subset, seen, unseen, dataset, name, x, y in episode:
        size[subset] += 1
        datasets.append(dataset)
        names.append(name)
        xs.append(x)
        ys.append(y)
    x = torch.stack(xs)
    y = torch.stack(ys)

    episode = {
        'n_trn': size[TRN_IDX],
        'n_tst': size[TST_IDX],
        'seen': seen,
        'unseen': unseen,
        'dataset': datasets,
        'name': names,
        'x': x,
        'y': y,
    }
    return episode


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dl(mset, batch_size, hparams):
    """Builds a meta dataloader for XRayMetaDatatset episode sampling.

    Parameters
    ----------
    mset : {'mtrn', 'mval', 'mtst'}
        Meta-dataset to load.
    hparams : SimpleNamespace
        data_aug : bool
            Enable data augmentation.
        num_workers : int
            Number of process for the dataloader.
        seed : bool
            Seed to init generators.
        debug : bool, default=True
            If True, prints loading info.

    Returns
    -------
    DataLoader
        The meta dataloader.
    """

    tsfm = build_tsfm(hparams.data_aug, hparams, hparams.debug)

    dataset = XRayDataset(mset, tsfm, hparams)

    g = torch.Generator()
    g.manual_seed(hparams.seed)

    dl = DataLoader(
        dataset=dataset,
        collate_fn=collate_batch,
        shuffle=True,
        batch_size=batch_size,
        num_workers=hparams.num_workers,
        worker_init_fn=seed_worker,
        generator=g
    )

    return dl


def build_mdl(mset, n_episodes, n_way, n_unseen, trn_k_shot, tst_kshot, hparams):
    """Builds a meta dataloader for XRayMetaDatatset episode sampling.

    Parameters
    ----------
    mset : {'mtrn', 'mval', 'mtst'}
        Meta-dataset to load.
    n_episodes : int
        Number of episodes.
    n_way : int
        Number of classes per episode.
    n_unseen : int
        Number of unseen classes per episode.
    trn_k_shot : int
        Minimal number of examples per classes for episode in training.
    tst_k_shot : int
        Minimal number of examples per classes for episode in testing.
    hparams : SimpleNamespace
        distro : str
            Distribution name.
        num_workers : int
            Number of process for the dataloader.
        debug : bool, default=True
            If True, prints loading info.

    Returns
    -------
    DataLoader
        The meta dataloader.
    """

    trn_tsfm = build_tsfm(hparams.data_aug, hparams, hparams.debug)
    tst_tsfm = build_tsfm(False, hparams, hparams.debug)

    dataset = XRayMetaDatatset(mset, trn_tsfm, tst_tsfm, hparams)

    sampler = EpisodeSampler(
        dataset, n_episodes, n_way, n_unseen, trn_k_shot, tst_kshot)

    g = torch.Generator()
    g.manual_seed(hparams.seed)

    mdl = DataLoader(
        dataset, batch_sampler=sampler,
        collate_fn=collate_episode,
        num_workers=hparams.num_workers,
        worker_init_fn=seed_worker,
        generator=g
    )

    return mdl


def show_grid(x):
    import matplotlib.pyplot as plt
    import numpy as np

    from torchvision.utils import make_grid

    grid = make_grid(x, value_range=(0, 255))
    grid = np.array(F.to_pil_image(grid.detach()))

    plt.imshow(grid)
    plt.show()


def test_build_dl(
        mset='mtrn',
        image_size=384, data_aug=False,
        norm={'mean': [0.0, 0.0, 0.0], 'std': [1.0, 1.0, 1.0]},
        batch_size=64,
        num_workers=0,
        batches=1,
        data_distro='complete',
        seed=0,
        debug=True):

    hparams = locals()

    from itertools import islice as take
    from pprint import pprint
    from types import SimpleNamespace

    hparams = SimpleNamespace(**hparams)

    dl = build_dl(mset, batch_size, hparams)

    print(
        'Dataloader:\n'
        f'  number of examples: {len(dl.dataset)}\n'
        f'          batch size: {batch_size}\n'
        f'   number of batches: {len(dl)}\n'
    )

    for batch in take(dl, batches):
        unseen = batch['unseen']
        seen = batch['seen']
        dataset = batch['dataset']
        name = batch['name']
        x = batch['x']
        y = batch['y']

        print(f'x shape={x.shape} dtype={x.dtype} '
              f'mean={x.type(torch.float).mean().round(decimals=2)} '
              f'min={x.min()} max={x.max()}')
        print(f'y shape={y.shape} dtype={y.dtype}')
        print(seen, unseen)

        datasets = np.array(dataset)
        names = np.array(name)
        data = np.column_stack([datasets, names, y.type(torch.int)])
        cols = ['dataset', 'name'] + unseen + seen
        df = pd.DataFrame(data, columns=cols)
        pd.set_option('display.max_colwidth', 500)
        print(df)

        if debug:
            show_grid(x)


def test_build_mdl(
        mset='mtrn',
        image_size=384, data_aug=False,
        n_episodes=1, n_way=3, n_unseen=1,
        trn_k_shot=30, tst_k_shot=30,
        norm={'mean': [0.0, 0.0, 0.0], 'std': [1.0, 1.0, 1.0]},
        num_workers=0,
        data_distro='complete',
        data_complete_with_norm='1',
        seed=0,
        debug=True):

    hparams = locals()

    from itertools import islice as take
    from types import SimpleNamespace

    hparams = SimpleNamespace(**hparams)

    np.random.seed(seed)
    random.seed(seed)

    mdl = build_mdl(
        mset, n_episodes, n_way, n_unseen, trn_k_shot, tst_k_shot, hparams)

    episode_size = []

    for episode in take(mdl, n_episodes):
        seen = episode['seen']
        unseen = episode['unseen']
        n_trn = episode['n_trn']
        n_tst = episode['n_tst']
        dataset = episode['dataset']
        name = episode['name']
        x = episode['x']
        y = episode['y']

        print(f'seen={seen} unseen={unseen}')
        print(f'n_trn={n_trn} n_tst={n_tst}')
        print(f'name shape={len(name)}')
        print(f'x shape={x.shape} dtype={x.dtype} '
              f'mean={x.type(torch.float).mean().round(decimals=2)} '
              f'min={x.min()} max={x.max()}')
        print(f'y shape={y.shape} dtype={y.dtype}')

        subset = n_trn * ['trn'] + n_tst * ['tst']
        datasets = np.array(dataset)
        names = np.array(name)
        data = np.column_stack([subset, datasets, names, y.type(torch.int)])
        cols = ['subset', 'dataset', 'name'] + seen + unseen
        df = pd.DataFrame(data, columns=cols)
        #with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)
        df.to_csv('test_build_mdl.csv')

        if debug:
            show_grid(x)

        episode_size.append(n_trn + n_tst)

    print(f'Mean episode size: {np.mean(episode_size)}')


def compute_mean_mdl(
        mset='mtrn',
        image_size=384, data_aug=False,
        n_episodes=1000, n_way=3, n_unseen=1,
        trn_k_shot=5, tst_k_shot=15,
        norm={'mean': [0.0, 0.0, 0.0], 'std': [1.0, 1.0, 1.0]},
        num_workers=0,
        data_distro='complete',
        seed=0,
        debug=True):

    hparams = locals()

    from types import SimpleNamespace
    from tqdm import tqdm

    hparams = SimpleNamespace(**hparams)

    mdl = build_mdl(
        mset, n_episodes, n_way, n_unseen, trn_k_shot, tst_k_shot, hparams)

    episode_sizes = []
    for episode in tqdm(mdl):
        episode_sizes.append(episode['n_trn'] + episode['n_tst'])

    total = len(mdl.dataset)
    episode_size = np.mean(episode_sizes)
    examples_used = np.sum(episode_sizes)


    print(
        'Dataloader:\n'
        f'      total examples: {total}\n'
        f'        episode size: {episode_size}\n'
        f'   number of batches: {len(mdl)}\n'
        f'       examples used: {examples_used}\n'
        f'     proportion used: {examples_used / total}\n'
        f'           eval freq: {total / examples_used}\n'
    )


if __name__ == '__main__':
    import fire
    fire.Fire()
