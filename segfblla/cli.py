#! /usr/bin/env python
import logging

import click
from PIL import Image
from rich.traceback import install

from kraken.lib import log

from segfblla import train

def set_logger(logger=None, level=logging.ERROR):
    logger.addHandler(RichHandler(rich_tracebacks=True))
    logger.setLevel(level)

# raise default max image size to 20k * 20k pixels
Image.MAX_IMAGE_PIXELS = 20000 ** 2

logging.captureWarnings(True)
logger = logging.getLogger()

APP_NAME = 'segfblla'

logging.captureWarnings(True)
logger = logging.getLogger('segfblla')

# install rich traceback handler
install(suppress=[click])

# raise default max image size to 20k * 20k pixels
Image.MAX_IMAGE_PIXELS = 20000 ** 2

@click.group()
@click.version_option()
@click.pass_context
@click.option('-v', '--verbose', default=0, count=True)
@click.option('-s', '--seed', default=None, type=click.INT,
              help='Seed for numpy\'s and torch\'s RNG. Set to a fixed value to '
                   'ensure reproducible random splits of data')
@click.option('-r', '--deterministic/--no-deterministic', default=False,
              help="Enables deterministic training. If no seed is given and enabled the seed will be set to 42.")
def cli(ctx, verbose, seed, deterministic):
    ctx.meta['deterministic'] = False if not deterministic else 'warn'
    if seed:
        from pytorch_lightning import seed_everything
        seed_everything(seed, workers=True)
    elif deterministic:
        from pytorch_lightning import seed_everything
        seed_everything(42, workers=True)

    ctx.meta['verbose'] = verbose
    log.set_logger(logger, level=30 - min(10 * verbose, 20))

cli.add_command(train.segtrain)

if __name__ == '__main__':
    cli()
