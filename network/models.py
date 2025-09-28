from os.path import isfile, join

import timm
from timm.models.helpers import load_checkpoint

from .backbones import BACKBONES


def _create_with_timm(timm_name, weights, features_only):
    pretrained = weights != 'random'
    return timm.create_model(
        model_name=timm_name,
        pretrained=pretrained,
        num_classes=0,
        features_only=features_only
    )


def _create_from_checkpoint(timm_name, weights, features_only, checkpoints_dir):
    path = join(checkpoints_dir, weights)
    if not isfile(path):
        raise FileNotFoundError(f'Checkpoint not found {path}')
    model = timm.create_model(
        model_name=timm_name,
        pretrained=False,
        num_classes=0,
        features_only=features_only,
    )
    print(f"Loading checkpoint: {path}")
    incompatible_keys = load_checkpoint(model, path, strict=False)
    print(incompatible_keys)
    return model


def _create(model_name, timm_name, weights, features_only, checkpoints_dir):
    if not timm_name:
        raise NotImplementedError('Pretrained model not implemented '
                                  f'model={model_name} weights={weights}')
    if weights in {'random', 'i1k', 'i21k',
                   'mim_in22k_ft_in22k_in1k', 'mim_m38m_ft_in22k_in1k'}:
        return _create_with_timm(timm_name, weights, features_only)
    else:
        return _create_from_checkpoint(
            timm_name, weights, features_only, checkpoints_dir)


@BACKBONES.register('convnext-atto')
def convnext_atto(weights, features_only, checkpoints_dir):
    model_name = 'convnext-atto'
    timm_name = {
        'random': 'convnext_atto.d2_in1k',
        'i1k': 'convnext_atto.d2_in1k',
        'i21k': None,
    }.get(weights, 'convnext_atto.d2_in1k')
    model = _create(model_name, timm_name, weights, features_only, checkpoints_dir)
    model.out_features = 320
    return model


@BACKBONES.register('convnext-tiny')
def convnext_tiny(weights, features_only, checkpoints_dir):
    model_name = 'convnext-tiny'
    timm_name = {
        'random': 'convnext_tiny.fb_in1k',
        'i1k': 'convnext_tiny.fb_in1k',
        'i21k': 'convnext_tiny.fb_in22k',
    }.get(weights, 'convnext_tiny')
    model = _create(model_name, timm_name, weights, features_only, checkpoints_dir)
    model.out_features = 768
    return model


@BACKBONES.register('convnextv2-atto')
def convnextv2_atto(weights, features_only, checkpoints_dir):
    model_name = 'convnextv2-atto'
    timm_name = {
        'random': 'convnextv2_atto',
        'i1k': 'convnextv2_atto.fcmae_ft_in1k',
        'i21k': None,
    }.get(weights, 'convnextv2_atto')
    model = _create(model_name, timm_name, weights, features_only, checkpoints_dir)
    model.out_features = 320
    return model


@BACKBONES.register('convnextv2-nano')
def convnextv2_nano(weights, features_only, checkpoints_dir):
    model_name = 'convnextv2-nano'
    timm_name = {
        'random': 'convnextv2_nano',
        'i1k': 'convnextv2_nano.fcmae_ft_in1k',
        'i21k': 'convnextv2_nano.fcmae_ft_in22k_in1k_384',
    }.get(weights, 'convnextv2_nano')
    model = _create(model_name, timm_name, weights, features_only, checkpoints_dir)
    model.out_features = 640
    return model


@BACKBONES.register('convnextv2-tiny')
def convnextv2_tiny(weights, features_only, checkpoints_dir):
    model_name = 'convnextv2-tiny'
    timm_name = {
        'random': 'convnextv2_tiny',
        'i1k': 'convnextv2_tiny.fcmae_ft_in1k',
        'i21k': 'convnextv2_tiny.fcmae_ft_in22k_in1k_384',
    }.get(weights, 'convnextv2_tiny')
    model = _create(model_name, timm_name, weights, features_only, checkpoints_dir)
    model.out_features = 768
    return model


@BACKBONES.register('densenet121')
def densenet121(weights, features_only, checkpoints_dir):
    model_name = 'densenet121'
    timm_name = {
        'random': 'densenet121.ra_in1k',
        'i1k': 'densenet121.ra_in1k',
        'i21k': None,
    }.get(weights, 'densenet121.ra_in1k')
    model = _create(model_name, timm_name, weights, features_only, checkpoints_dir)
    model.out_features = 1024
    return model


@BACKBONES.register('densenet161')
def densenet161(weights, features_only, checkpoints_dir):
    model_name = 'densenet161'
    timm_name = {
        'random': 'densenet161.tv_in1k',
        'i1k': 'densenet161.tv_in1k',
        'i21k': None,
    }.get(weights, 'densenet161.tv_in1k')
    model = _create(model_name, timm_name, weights, features_only, checkpoints_dir)
    model.out_features = 2208
    return model


@BACKBONES.register('eva02-large')
def eva02_large(weights, features_only, checkpoints_dir):
    model_name = 'eva02_large_patch14_448'
    timm_name = {
        'random': 'eva02_large_patch14_448.mim_m38m_ft_in22k_in1k',
        'i1k': None,
        'mim_in22k_ft_in22k_in1k': 'eva02_large_patch14_448.mim_in22k_ft_in22k_in1k',
        'mim_m38m_ft_in22k_in1k': 'eva02_large_patch14_448.mim_m38m_ft_in22k_in1k',
    }.get(weights, 'eva02_large_patch14_448.mim_m38m_ft_in22k_in1k')
    model = _create(model_name, timm_name, weights, features_only, checkpoints_dir)
    model.out_features = 1024
    return model


@BACKBONES.register('eva02-small')
def eva02_small(weights, features_only, checkpoints_dir):
    model_name = 'eva02_small'
    timm_name = {
        'random': 'eva02_small_patch14_336.mim_in22k_ft_in1k',
        'i1k': None,
        'i21k': 'eva02_small_patch14_336.mim_in22k_ft_in1k',
    }.get(weights, 'eva02_small_patch14_336.mim_in22k_ft_in1k')
    model = _create(model_name, timm_name, weights, features_only, checkpoints_dir)
    model.out_features = 384
    return model


@BACKBONES.register('eva02-tiny')
def eva02_tiny(weights, features_only, checkpoints_dir):
    model_name = 'eva02_tiny'
    timm_name = {
        'random': 'eva02_tiny_patch14_336.mim_in22k_ft_in1k',
        'i1k': None,
        'i21k': 'eva02_tiny_patch14_336.mim_in22k_ft_in1k',
    }.get(weights, 'eva02_tiny_patch14_336.mim_in22k_ft_in1k')
    model = _create(model_name, timm_name, weights, features_only, checkpoints_dir)
    model.out_features = 192
    return model


@BACKBONES.register('mobilenetv3-large-100')
def mobilenetv3_large_100(weights, features_only, checkpoints_dir):
    model_name = 'mobilenetv3-large-100'
    timm_name = {
        'random': 'mobilenetv3_large_100.miil_in21k',
        'i1k': 'tf_mobilenetv3_large_100.in1k',
        'i21k': 'mobilenetv3_large_100.miil_in21k',
    }.get(weights, 'mobilenetv3_large_100.miil_in21k')
    model = _create(model_name, timm_name, weights, features_only, checkpoints_dir)
    model.out_features = 1280
    return model


@BACKBONES.register('mobilenetv3-small-075')
def mobilenetv3_small_075(weights, features_only, checkpoints_dir):
    model_name = 'mobilenetv3-small-075'
    timm_name = {
        'random': 'tf_mobilenetv3_small_075.in1k',
        'i1k': 'tf_mobilenetv3_small_075.in1k',
        'i21k': None,
    }.get(weights, 'tf_mobilenetv3_small_075.in1k')
    model = _create(model_name, timm_name, weights, features_only, checkpoints_dir)
    model.out_features = 1024
    return model


@BACKBONES.register('mobilevitv2-050')
def mobilevitv2_050(weights, features_only, checkpoints_dir):
    model_name = 'mobilevitv2-050'
    timm_name = {
        'random': 'mobilevitv2_050.cvnets_in1k',
        'i1k': 'mobilevitv2_050.cvnets_in1k',
        'i21k': None,
    }.get(weights, 'mobilevitv2_050.cvnets_in1k')
    model = _create(model_name, timm_name, weights, features_only, checkpoints_dir)
    model.out_features = 256
    return model


@BACKBONES.register('mobilevitv2-100')
def mobilevitv2_100(weights, features_only, checkpoints_dir):
    model_name = 'mobilevitv2-100'
    timm_name = {
        'random': 'mobilevitv2_100.cvnets_in1k',
        'i1k': 'mobilevitv2_100.cvnets_in1k',
        'i21k': None,
    }.get(weights, 'mobilevitv2_100.cvnets_in1k')
    model = _create(model_name, timm_name, weights, features_only, checkpoints_dir)
    model.out_features = 512
    return model


@BACKBONES.register('mobilevitv2-200')
def mobilevitv2_200(weights, features_only, checkpoints_dir):
    model_name = 'mobilevitv2-200'
    timm_name = {
        'random': 'mobilevitv2_200.cvnets_in1k',
        'i1k': 'mobilevitv2_200.cvnets_in1k',
        'i21k': 'mobilevitv2_200.cvnets_in22k_ft_in1k_384',
    }.get(weights, 'mobilevitv2_200.cvnets_in1k')
    model = _create(model_name, timm_name, weights, features_only, checkpoints_dir)
    model.out_features = 1024
    return model


__all__ = [
    'convnext_atto', 'convnext_tiny',
    'convnextv2_atto', 'convnextv2_nano', 'convnextv2_tiny',
    'densenet121', 'densenet161',
    'eva02_large', 'eva02_small', 'eva02_tiny',
    'mobilenetv3_large_100', 'mobilenetv3_small_075',
    'mobilevitv2_050', 'mobilevitv2_100', 'mobilevitv2_200',
]
