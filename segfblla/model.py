#
# Copyright 2015 Benjamin Kiessling
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
"""
Training loop interception helpers
"""
import logging
import re
import warnings
from typing import (TYPE_CHECKING, Any, Callable, Dict, Literal, Optional,
                    Sequence, Union)

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from pytorch_lightning.callbacks import Callback, EarlyStopping
from torch.optim import lr_scheduler
from torchmetrics.classification import MultilabelAccuracy, MultilabelJaccardIndex

from transformers import SegformerForSemanticSegmentation


from segfblla import default_specs
from segfblla.losses import GeneralizedDiceLoss
from kraken.lib.xml import XMLPage

if TYPE_CHECKING:
    from os import PathLike

logger = logging.getLogger(__name__)


class SegmentationModel(pl.LightningModule):
    def __init__(self,
                 num_classes: int,
                 hyper_params: Dict = None,
                 batches_per_epoch: int = 0):
        """
        A LightningModule encapsulating the training setup for a page
        segmentation model.

        Setup parameters (load, training_data, evaluation_data, ....) are
        named, model hyperparameters (everything in
        `kraken.lib.default_specs.SEGMENTATION_HYPER_PARAMS`) are in in the
        `hyper_params` argument.

        Args:
            hyper_params (dict): Hyperparameter dictionary containing all fields
                                 from
                                 kraken.lib.default_specs.SEGMENTATION_HYPER_PARAMS
            **kwargs: Setup parameters, i.e. CLI parameters of the segtrain() command.
        """
        super().__init__()

        self.best_epoch = -1
        self.best_metric = 0.0
        self.best_model = None

        hyper_params_ = default_specs.SEGMENTATION_HYPER_PARAMS.copy()

        if hyper_params:
            hyper_params_.update(hyper_params)

        self.save_hyperparameters()

        # set multiprocessing tensor sharing strategy
        if 'file_system' in torch.multiprocessing.get_all_sharing_strategies():
            logger.debug('Setting multiprocessing tensor sharing strategy to file_system')
            torch.multiprocessing.set_sharing_strategy('file_system')

        logger.info(f'Creating segformer model with {num_classes} outputs')
        self.net = SegformerForSemanticSegmentation.from_pretrained('nvidia/mit-b0',
                                                                    num_labels=num_classes)
        self.net = self.net.train()
        self.model_config = self.net.config.to_dict()

        # loss
        if hyper_params['loss'] == 'bce':
            self.criterion = nn.BCEWithLogitsLoss()
        elif hyper_params['loss'] == 'gdl':
            self.criterion = GeneralizedDiceLoss()
        else:
            raise ValueError(f'Unknown loss {hyper_params["loss"]} in hyperparameter dict')

        self.val_px_accuracy = MultilabelAccuracy(average='micro', num_labels=num_classes)
        self.val_mean_accuracy = MultilabelAccuracy(average='macro', num_labels=num_classes)
        self.val_mean_iu = MultilabelJaccardIndex(average='macro', num_labels=num_classes)
        self.val_freq_iu = MultilabelJaccardIndex(average='weighted', num_labels=num_classes)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        input, target = batch['image'], batch['target']
        output = self.net(input)
        output = F.interpolate(output.logits, size=(target.size(2), target.size(3)))
        loss = self.criterion(output, target)
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['target']
        pred = self.net(x)
        pred = torch.sigmoid(pred.logits)
        # scale target to output size
        y = F.interpolate(y, size=(pred.size(2), pred.size(3))).int()

        self.val_px_accuracy.update(pred, y)
        self.val_mean_accuracy.update(pred, y)
        self.val_mean_iu.update(pred, y)
        self.val_freq_iu.update(pred, y)

    def on_validation_epoch_end(self):

        pixel_accuracy = self.val_px_accuracy.compute()
        mean_accuracy = self.val_mean_accuracy.compute()
        mean_iu = self.val_mean_iu.compute()
        freq_iu = self.val_freq_iu.compute()

        if mean_iu > self.best_metric:
            logger.debug(f'Updating best metric from {self.best_metric} ({self.best_epoch}) to {mean_iu} ({self.current_epoch})')
            self.best_epoch = self.current_epoch
            self.best_metric = mean_iu

        logger.info(f'validation run: accuracy {pixel_accuracy} mean_acc {mean_accuracy} mean_iu {mean_iu} freq_iu {freq_iu}')

        self.log('val_accuracy', pixel_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_mean_acc', mean_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_mean_iu', mean_iu, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_freq_iu', freq_iu, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_metric', mean_iu, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        self.val_px_accuracy.reset()
        self.val_mean_accuracy.reset()
        self.val_mean_iu.reset()
        self.val_freq_iu.reset()

    def on_save_checkpoint(self, checkpoint):
        checkpoint['model_config'] = self.model_config

    def on_load_checkpoint(self, checkpoint):
        self.model_config = checkpoint['model_config']

    def save_checkpoint(self, filename):
        self.trainer.save_checkpoint(filename)

    def configure_callbacks(self):
        callbacks = []
        if self.hparams.hyper_params['quit'] == 'early':
            callbacks.append(EarlyStopping(monitor='val_mean_iu',
                                           mode='max',
                                           patience=self.hparams.hyper_params['lag'],
                                           stopping_threshold=1.0))

        return callbacks

    # configuration of optimizers and learning rate schedulers
    # --------------------------------------------------------
    #
    # All schedulers are created internally with a frequency of step to enable
    # batch-wise learning rate warmup. In lr_scheduler_step() calls to the
    # scheduler are then only performed at the end of the epoch.
    def configure_optimizers(self):
        return _configure_optimizer_and_lr_scheduler(self.hparams.hyper_params,
                                                     self.net.parameters(),
                                                     len_train_set=self.hparams.batches_per_epoch,
                                                     loss_tracking_mode='max')

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # update params
        optimizer.step(closure=optimizer_closure)

        # linear warmup between 0 and the initial learning rate `lrate` in `warmup`
        # steps.
        if self.hparams.hyper_params['warmup'] and self.trainer.global_step < self.hparams.hyper_params['warmup']:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.hparams.hyper_params['warmup'])
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.hyper_params['lrate']

    def lr_scheduler_step(self, scheduler, metric):
        if not self.hparams.hyper_params['warmup'] or self.trainer.global_step >= self.hparams.hyper_params['warmup']:
            # step OneCycleLR each batch if not in warmup phase
            if isinstance(scheduler, lr_scheduler.OneCycleLR):
                scheduler.step()
            # step every other scheduler epoch-wise
            elif self.trainer.is_last_batch:
                if metric is None:
                    scheduler.step()
                else:
                    scheduler.step(metric)


def _configure_optimizer_and_lr_scheduler(hparams, params, len_train_set=None, loss_tracking_mode='max'):
    optimizer = hparams.get("optimizer")
    lrate = hparams.get("lrate")
    momentum = hparams.get("momentum")
    weight_decay = hparams.get("weight_decay")
    schedule = hparams.get("schedule")
    gamma = hparams.get("gamma")
    step_size = hparams.get("step_size")
    rop_factor = hparams.get("rop_factor")
    rop_patience = hparams.get("rop_patience")
    epochs = hparams.get("epochs")
    completed_epochs = hparams.get("completed_epochs")

    # XXX: Warmup is not configured here because it needs to be manually done in optimizer_step()
    logger.debug(f'Constructing {optimizer} optimizer (lr: {lrate}, momentum: {momentum})')
    if optimizer in ['Adam', 'AdamW']:
        optim = getattr(torch.optim, optimizer)(params, lr=lrate, weight_decay=weight_decay)
    else:
        optim = getattr(torch.optim, optimizer)(params,
                                                lr=lrate,
                                                momentum=momentum,
                                                weight_decay=weight_decay)
    lr_sched = {}
    if schedule == 'exponential':
        lr_sched = {'scheduler': lr_scheduler.ExponentialLR(optim, gamma, last_epoch=completed_epochs-1),
                    'interval': 'step'}
    elif schedule == 'cosine':
        lr_sched = {'scheduler': lr_scheduler.CosineAnnealingLR(optim, gamma, last_epoch=completed_epochs-1),
                    'interval': 'step'}
    elif schedule == 'step':
        lr_sched = {'scheduler': lr_scheduler.StepLR(optim, step_size, gamma, last_epoch=completed_epochs-1),
                    'interval': 'step'}
    elif schedule == 'reduceonplateau':
        lr_sched = {'scheduler': lr_scheduler.ReduceLROnPlateau(optim,
                                                                mode=loss_tracking_mode,
                                                                factor=rop_factor,
                                                                patience=rop_patience),
                    'interval': 'step'}
    elif schedule == '1cycle':
        if epochs <= 0:
            raise ValueError('1cycle learning rate scheduler selected but '
                             'number of epochs is less than 0 '
                             f'({epochs}).')
        last_epoch = completed_epochs*len_train_set if completed_epochs else -1
        lr_sched = {'scheduler': lr_scheduler.OneCycleLR(optim,
                                                         max_lr=lrate,
                                                         epochs=epochs,
                                                         steps_per_epoch=len_train_set,
                                                         last_epoch=last_epoch),
                    'interval': 'step'}
    elif schedule != 'constant':
        raise ValueError(f'Unsupported learning rate scheduler {schedule}.')

    ret = {'optimizer': optim}
    if lr_sched:
        ret['lr_scheduler'] = lr_sched

    if schedule == 'reduceonplateau':
        lr_sched['monitor'] = 'val_mean_iu'
        lr_sched['strict'] = False
        lr_sched['reduce_on_plateau'] = True

    return ret
