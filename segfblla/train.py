#
# Copyright 2022 Benjamin Kiessling
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
segfbla.train
~~~~~~~~~~~~~

Command line driver for segmentation training and evaluation.
"""
import logging
import pathlib
import shlex
from typing import Dict

import click
from PIL import Image

from kraken.ketos.util import (_expand_gt, _validate_manifests, message,
                               to_ptl_device)
from kraken.lib.exceptions import KrakenInputException

from segfblla.default_specs import SEGMENTATION_HYPER_PARAMS

logging.captureWarnings(True)
logger = logging.getLogger('kraken')

# raise default max image size to 20k * 20k pixels
Image.MAX_IMAGE_PIXELS = 20000 ** 2


def _validate_merging(ctx, param, value):
    """
    Maps baseline/region merging to a dict of merge structures.
    """
    if not value:
        return None
    merge_dict: Dict[str, str] = {}
    try:
        for m in value:
            lexer = shlex.shlex(m, posix=True)
            lexer.wordchars += r'\/.+-()=^&;,.'
            tokens = list(lexer)
            if len(tokens) != 3:
                raise ValueError
            k, _, v = tokens
            merge_dict[v] = k  # type: ignore
    except Exception:
        raise click.BadParameter('Mappings must be in format target:src')
    return merge_dict


@click.command('segtrain')
@click.pass_context
@click.option('-B', '--batch-size', show_default=True, type=click.INT,
              default=SEGMENTATION_HYPER_PARAMS['batch_size'], help='batch sample size')
@click.option('-o', '--output', show_default=True, type=click.Path(), default='model', help='Output model file')
@click.option('--line-width',
              show_default=True,
              default=SEGMENTATION_HYPER_PARAMS['line_width'],
              help='The height of each baseline in the target after scaling')
@click.option('-i', '--load', show_default=True, type=click.Path(exists=True,
              readable=True), help='Load existing file to continue training')
@click.option('-F', '--freq', show_default=True, default=SEGMENTATION_HYPER_PARAMS['freq'], type=click.FLOAT,
              help='Model saving and report generation frequency in epochs '
                   'during training. If frequency is >1 it must be an integer, '
                   'i.e. running validation every n-th epoch.')
@click.option('-q',
              '--quit',
              show_default=True,
              default=SEGMENTATION_HYPER_PARAMS['quit'],
              type=click.Choice(['early',
                                 'fixed']),
              help='Stop condition for training. Set to `early` for early stopping or `fixed` for fixed number of epochs')
@click.option('-N',
              '--epochs',
              show_default=True,
              default=SEGMENTATION_HYPER_PARAMS['epochs'],
              help='Number of epochs to train for')
@click.option('--min-epochs',
              show_default=True,
              default=SEGMENTATION_HYPER_PARAMS['min_epochs'],
              help='Minimal number of epochs to train for when using early stopping.')
@click.option('--lag',
              show_default=True,
              default=SEGMENTATION_HYPER_PARAMS['lag'],
              help='Number of evaluations (--report frequency) to wait before stopping training without improvement')
@click.option('--min-delta',
              show_default=True,
              default=SEGMENTATION_HYPER_PARAMS['min_delta'],
              type=click.FLOAT,
              help='Minimum improvement between epochs to reset early stopping. By default it scales the delta by the best loss')
@click.option('-d', '--device', show_default=True, default='cpu', help='Select device to use (cpu, cuda:0, cuda:1, ...)')
@click.option('--precision',
              show_default=True,
              default='16',
              type=click.Choice(['64', '32', 'bf16', '16']),
              help='Numerical precision to use for training. Default is 32-bit single-point precision.')
@click.option('--optimizer',
              show_default=True,
              default=SEGMENTATION_HYPER_PARAMS['optimizer'],
              type=click.Choice(['AdamW',
                                 'Adam',
                                 'SGD',
                                 'RMSprop',
                                 'Lamb']),
              help='Select optimizer')
@click.option('-r', '--lrate', show_default=True, default=SEGMENTATION_HYPER_PARAMS['lrate'], help='Learning rate')
@click.option('-m', '--momentum', show_default=True, default=SEGMENTATION_HYPER_PARAMS['momentum'], help='Momentum')
@click.option('-w', '--weight-decay', show_default=True,
              default=SEGMENTATION_HYPER_PARAMS['weight_decay'], help='Weight decay')
@click.option('--warmup', show_default=True, type=float,
              default=SEGMENTATION_HYPER_PARAMS['warmup'], help='Number of steps to ramp up to `lrate` initial learning rate.')
@click.option('--schedule',
              show_default=True,
              type=click.Choice(['constant',
                                 '1cycle',
                                 'exponential',
                                 'cosine',
                                 'step',
                                 'reduceonplateau']),
              default=SEGMENTATION_HYPER_PARAMS['schedule'],
              help='Set learning rate scheduler. For 1cycle, cycle length is determined by the `--step-size` option.')
@click.option('-g',
              '--gamma',
              show_default=True,
              default=SEGMENTATION_HYPER_PARAMS['gamma'],
              help='Decay factor for exponential, step, and reduceonplateau learning rate schedules')
@click.option('-ss',
              '--step-size',
              show_default=True,
              default=SEGMENTATION_HYPER_PARAMS['step_size'],
              help='Number of validation runs between learning rate decay for exponential and step LR schedules')
@click.option('--sched-patience',
              show_default=True,
              default=SEGMENTATION_HYPER_PARAMS['rop_patience'],
              help='Minimal number of validation runs between LR reduction for reduceonplateau LR schedule.')
@click.option('--cos-max',
              show_default=True,
              default=SEGMENTATION_HYPER_PARAMS['cos_t_max'],
              help='Epoch of minimal learning rate for cosine LR scheduler.')
@click.option('-p', '--partition', show_default=True, default=0.9,
              help='Ground truth data partition ratio between train/validation set')
@click.option('-t', '--training-files', show_default=True, default=None, multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with additional paths to training data')
@click.option('-e', '--evaluation-files', show_default=True, default=None, multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with paths to evaluation data. Overrides the `-p` parameter')
@click.option('--workers', show_default=True, default=1, type=click.IntRange(1), help='Number of worker proesses.')
@click.option('--threads', show_default=True, default=1, type=click.IntRange(1), help='Maximum size of OpenMP/BLAS thread pool.')
@click.option('--load-hyper-parameters/--no-load-hyper-parameters', show_default=True, default=False,
              help='When loading an existing model, retrieve hyper-parameters from the model')
@click.option('--force-binarization/--no-binarization', show_default=True,
              default=False, help='Forces input images to be binary, otherwise '
              'the appropriate color format will be auto-determined through the '
              'network specification. Will be ignored in `path` mode.')
@click.option('-f', '--format-type', type=click.Choice(['path', 'xml', 'alto', 'page']), default='xml',
              help='Sets the training data format. In ALTO and PageXML mode all '
              'data is extracted from xml files containing both baselines and a '
              'link to source images. In `path` mode arguments are image files '
              'sharing a prefix up to the last extension with JSON `.path` files '
              'containing the baseline information.')
@click.option('--suppress-regions/--no-suppress-regions', show_default=True,
              default=False, help='Disables region segmentation training.')
@click.option('--suppress-baselines/--no-suppress-baselines', show_default=True,
              default=False, help='Disables baseline segmentation training.')
@click.option('-vr', '--valid-regions', show_default=True, default=None, multiple=True,
              help='Valid region types in training data. May be used multiple times.')
@click.option('-vb', '--valid-baselines', show_default=True, default=None, multiple=True,
              help='Valid baseline types in training data. May be used multiple times.')
@click.option('-mr',
              '--merge-regions',
              show_default=True,
              default=None,
              help='Region merge mapping. One or more mappings of the form `$target:$src` where $src is merged into $target.',
              multiple=True,
              callback=_validate_merging)
@click.option('-mb',
              '--merge-baselines',
              show_default=True,
              default=None,
              help='Baseline type merge mapping. Same syntax as `--merge-regions`',
              multiple=True,
              callback=_validate_merging)
@click.option('-br', '--bounding-regions', show_default=True, default=None, multiple=True,
              help='Regions treated as boundaries for polygonization purposes. May be used multiple times.')
@click.option('--augment/--no-augment',
              show_default=True,
              default=SEGMENTATION_HYPER_PARAMS['augment'],
              help='Enable image augmentation')
@click.option('--resize', show_default=True, default='fail',
              type=click.Choice([
                  'add', 'union',  # Deprecation: `add` is deprecated, `union` is the new value
                  'both', 'new',  # Deprecation: `both` is deprecated, `new` is the new value
                  'fail'
              ]),
              help='Output layer resizing option. If set to `add` new classes will be '
                   'added, `both` will set the layer to match exactly '
                   'the training data classes, `fail` will abort if training data and model '
                   'classes do not match.')
@click.option('-tl', '--topline', 'topline', show_default=True, flag_value='topline',
              help='Switch for the baseline location in the scripts. '
                   'Set to topline if the data is annotated with a hanging baseline, as is '
                   'common with Hebrew, Bengali, Devanagari, etc. Set to '
                   ' centerline for scripts annotated with a central line.')
@click.option('-cl', '--centerline', 'topline', flag_value='centerline')
@click.option('-bl', '--baseline', 'topline', flag_value='baseline', default='baseline')
@click.option('--logger', 'pl_logger', show_default=True, type=click.Choice(['tensorboard']), default=None,
              help='Logger used by PyTorch Lightning to track metrics such as loss and accuracy.')
@click.option('--log-dir', show_default=True, type=click.Path(exists=True, dir_okay=True, writable=True),
              help='Path to directory where the logger will store the logs. If not set, a directory will be created in the current working directory.')
@click.argument('ground_truth', nargs=-1, callback=_expand_gt, type=click.Path(exists=False, dir_okay=False))
def segtrain(ctx, batch_size, output, line_width, load, freq, quit,
             epochs, min_epochs, lag, min_delta, device, precision, optimizer,
             lrate, momentum, weight_decay, warmup, schedule, gamma, step_size,
             sched_patience, cos_max, partition, training_files,
             evaluation_files, workers, threads, load_hyper_parameters,
             force_binarization, format_type, suppress_regions,
             suppress_baselines, valid_regions, valid_baselines, merge_regions,
             merge_baselines, bounding_regions, augment, resize, topline,
             pl_logger, log_dir, ground_truth):
    """
    Trains a baseline labeling model for layout analysis
    """
    import shutil

    from threadpoolctl import threadpool_limits

    from kraken.lib.train import KrakenTrainer
    from segfblla.model import SegmentationModel

    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint

    if resize != 'fail' and not load:
        raise click.BadOptionUsage('resize', 'resize option requires loading an existing model')

    if not (0 <= freq <= 1) and freq % 1.0 != 0:
        raise click.BadOptionUsage('freq', 'freq needs to be either in the interval [0,1.0] or a positive integer.')

    if augment:
        try:
            import albumentations  # NOQA
        except ImportError:
            raise click.BadOptionUsage('augment', 'augmentation needs the `albumentations` package installed.')

    if pl_logger == 'tensorboard':
        try:
            import tensorboard  # NOQA
        except ImportError:
            raise click.BadOptionUsage('logger', 'tensorboard logger needs the `tensorboard` package installed.')

    if log_dir is None:
        log_dir = pathlib.Path.cwd()

    logger.info('Building ground truth set from {} document images'.format(len(ground_truth) + len(training_files)))

    # populate hyperparameters from command line args
    hyper_params = SEGMENTATION_HYPER_PARAMS.copy()
    hyper_params.update({'line_width': line_width,
                         'batch_size': batch_size,
                         'freq': freq,
                         'quit': quit,
                         'epochs': epochs,
                         'min_epochs': min_epochs,
                         'lag': lag,
                         'min_delta': min_delta,
                         'optimizer': optimizer,
                         'lrate': lrate,
                         'momentum': momentum,
                         'weight_decay': weight_decay,
                         'warmup': warmup,
                         'schedule': schedule,
                         'augment': augment,
                         'gamma': gamma,
                         'step_size': step_size,
                         'rop_patience': sched_patience,
                         'cos_t_max': cos_max,
                         })

    # disable automatic partition when given evaluation set explicitly
    if evaluation_files:
        partition = 1
    ground_truth = list(ground_truth)

    # merge training_files into ground_truth list
    if training_files:
        ground_truth.extend(training_files)

    if len(ground_truth) == 0:
        raise click.UsageError('No training data was provided to the train command. Use `-t` or the `ground_truth` argument.')

    loc = {'topline': True,
           'baseline': False,
           'centerline': None}

    topline = loc[topline]

    try:
        accelerator, device = to_ptl_device(device)
    except Exception as e:
        raise click.BadOptionUsage('device', str(e))

    if hyper_params['freq'] > 1:
        val_check_interval = {'check_val_every_n_epoch': int(hyper_params['freq'])}
    else:
        val_check_interval = {'val_check_interval': hyper_params['freq']}

    model = SegmentationModel(hyper_params,
                              output=output,
                              model=load,
                              training_data=ground_truth,
                              evaluation_data=evaluation_files,
                              partition=partition,
                              num_workers=workers,
                              load_hyper_parameters=load_hyper_parameters,
                              force_binarization=force_binarization,
                              format_type=format_type,
                              suppress_regions=suppress_regions,
                              suppress_baselines=suppress_baselines,
                              valid_regions=valid_regions,
                              valid_baselines=valid_baselines,
                              merge_regions=merge_regions,
                              merge_baselines=merge_baselines,
                              bounding_regions=bounding_regions,
                              resize=resize,
                              topline=topline)

    message('Training line types:')
    for k, v in model.train_set.dataset.class_mapping['baselines'].items():
        message(f'  {k}\t{v}\t{model.train_set.dataset.class_stats["baselines"][k]}')
    message('Training region types:')
    for k, v in model.train_set.dataset.class_mapping['regions'].items():
        message(f'  {k}\t{v}\t{model.train_set.dataset.class_stats["regions"][k]}')

    if len(model.train_set) == 0:
        raise click.UsageError('No valid training data was provided to the train command. Use `-t` or the `ground_truth` argument.')

    trainer = KrakenTrainer(accelerator=accelerator,
                            devices=device,
                            precision=precision,
                            max_epochs=hyper_params['epochs'] if hyper_params['quit'] == 'fixed' else -1,
                            min_epochs=hyper_params['min_epochs'],
                            enable_progress_bar=True if not ctx.meta['verbose'] else False,
                            deterministic=ctx.meta['deterministic'],
                            **val_check_interval)

    with threadpool_limits(limits=threads):
        trainer.fit(model)

    if model.best_epoch == -1:
        logger.warning('Model did not improve during training.')
        ctx.exit(1)

    if not model.current_epoch:
        logger.warning('Training aborted before end of first epoch.')
        ctx.exit(1)

    if quit == 'early':
        message(f'Moving best model {model.best_model} ({model.best_metric}) to {output}_best.mlmodel')
        logger.info(f'Moving best model {model.best_model} ({model.best_metric}) to {output}_best.mlmodel')
        shutil.copy(f'{model.best_model}', f'{output}_best.mlmodel')
