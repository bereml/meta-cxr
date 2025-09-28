""" factory.py """

import torch.nn as nn

from .backbones import BACKBONES


def freeze(module):
    for param in module.parameters():
        param.requires_grad = False

def unfreeze(module):
    for param in module.parameters():
        param.requires_grad = True

def freeze_and_eval(module, freeze_eval=True):
    if freeze_eval:
        freeze(module)
        module.eval()
    else:
        unfreeze(module)
        module.train()

def unfreeze_and_train(module):
    unfreeze(module)
    module.train()


class Net(nn.Module):

    def __init__(self, backbone, weights, features_only=False,
                 head_type='id', head_classes=0, checkpoints_dir='checkpoints'):
        super().__init__()
        if features_only:
            head_type, head_classes = 'id', 0
        self._create_backbone(backbone, weights, features_only, checkpoints_dir)
        self.new_head(head_type, head_classes)

    def _create_backbone(self, backbone, weights, features_only, checkpoints_dir):
        self.backbone_name = backbone
        self.backbone_weights = weights
        self.backbone_features_only = features_only
        create_fn = BACKBONES.get(backbone, None)
        if create_fn is None:
            raise NotImplementedError('Model not implemented: '
                                    f'backbone={backbone} weights={weights}')
        self.backbone = create_fn(weights, features_only, checkpoints_dir)
        if self.backbone is None:
            raise NotImplementedError('Model not implemented: '
                                    f'backbone={backbone} weights={weights}')

    def new_head(self, head_type, head_classes):
        self.head_type = head_type
        self.head_classes = head_classes
        if head_type == 'id':
            self.head = nn.Identity()
        elif head_classes == 0:
            raise ValueError(f'incompatible head_type={head_type} head_classes={head_classes}')
        elif head_type == 'avg':
            self.head = nn.AdaptiveAvgPool1d(head_classes)
        elif head_type == 'fc':
            self.head = nn.Linear(self.backbone.out_features, head_classes)
        else:
            raise ValueError(f'unknown head_type={head_type}')
        self.head.to(next(self.backbone.parameters()).device)

    def reset_head(self):
        self.head.reset_parameters()

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

    def freeze_and_eval_backbone(self, freeze_eval=True):
        freeze_and_eval(self.backbone, freeze_eval)

    def freeze_and_eval_head(self, freeze_eval=True):
        freeze_and_eval(self.head, freeze_eval)

    def unfreeze_and_train_backbone(self):
        unfreeze_and_train(self.backbone)

    def unfreeze_and_train_head(self):
        unfreeze_and_train(self.head)

    def extra_repr(self):
        return 'backbone={}, weights={}, features_only={}, head_type={}, head_type={}'.format(
            self.backbone_name, self.backbone_weights, self.backbone_features_only,
            self.head_type, self.head_type
        )

def create_net(backbone, weights='random', features_only=False,
               head_type='id', head_classes=0, checkpoints_dir='checkpoints'):
    return Net(backbone, weights, features_only,
               head_type, head_classes, checkpoints_dir)
