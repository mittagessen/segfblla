#
# Copyright 2024 Benjamin Kiessling
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
segfblla.pred
~~~~~~~~~~~~~

Command line drivers for recognition inference.
"""
import dataclasses
import logging
import os
import shlex
import warnings
from functools import partial
from pathlib import Path
from typing import IO, Any, Callable, Dict, List, Union, cast

import click
import importlib_resources
from PIL import Image
from rich.traceback import install

from .util import message

logging.captureWarnings(True)
logger = logging.getLogger('kraken')


@click.command('segment')
@click.pass_context
@click.version_option()
@click.option('-i', '--input',
              type=(click.Path(exists=True, dir_okay=False, path_type=Path),  # type: ignore
                    click.Path(writable=True, dir_okay=False, path_type=Path)),
              multiple=True,
              help='Input-output file pairs. Each input file (first argument) is mapped to one '
                   'output file (second argument), e.g. `-i input.png output.txt`')
@click.option('-I', '--batch-input', multiple=True, help='Glob expression to add multiple files at once.')
@click.option('-o', '--suffix', default='', show_default=True,
              help='Suffix for output files from batch and PDF inputs.')
@click.option('-f', '--format-type', type=click.Choice(['image', 'alto', 'page', 'pdf', 'xml']), default='image',
              help='Sets the default input type. In image mode inputs are image '
                   'files, alto/page expects XML files in the respective format, pdf '
                   'expects PDF files with numbered suffixes added to output file '
                   'names as needed.')
@click.option('-p', '--pdf-format', default='{src}_{idx:06d}',
              show_default=True,
              help='Format for output of PDF files. valid fields '
                   'are `src` (source file), `idx` (page number), and `uuid` (v4 uuid). '
                   '`-o` suffixes are appended to this format string.')
@click.option('-h', '--hocr', 'serializer',
              help='Switch between hOCR, ALTO, abbyyXML, PageXML or "native" '
              'output. Native are plain image files for image, JSON for '
              'segmentation, and text for transcription output.',
              flag_value='hocr')
@click.option('-a', '--alto', 'serializer', flag_value='alto')
@click.option('-y', '--abbyy', 'serializer', flag_value='abbyyxml')
@click.option('-x', '--pagexml', 'serializer', flag_value='pagexml')
@click.option('-n', '--native', 'serializer', flag_value='native', default=True,
              show_default=True)
@click.option('-t', '--template', type=click.Path(exists=True, dir_okay=False),
              help='Explicitly set jinja template for output serialization. Overrides -h/-a/-y/-x/-n.')
@click.option('-d', '--device', default='cpu', show_default=True,
              help='Select device to use (cpu, cuda:0, cuda:1, ...)')
@click.option('-r', '--raise-on-error/--no-raise-on-error', default=False, show_default=True,
              help='Raises the exception that caused processing to fail in the case of an error')
@click.option('--threads', default=1, show_default=True, type=click.IntRange(1),
              help='Size of thread pools for intra-op parallelization')
@click.option('-m', '--model', default=None, show_default=True,
              help='Baseline detection model to use')
@click.option('-d', '--text-direction', default='horizontal-lr',
              show_default=True,
              type=click.Choice(['horizontal-lr', 'horizontal-rl',
                                 'vertical-lr', 'vertical-rl']),
              help='Sets principal text direction')
def segment(input, batch_input, suffix, format_type, pdf_format,
            serializer, template, device, raise_on_error, threads,
            model, text_direction):
    """
    Segmentation inference with SegFormer models.

    Inputs are defined as one or more pairs `-i input_file output_file`
    followed by one or more chainable processing commands. Likewise, verbosity
    is set on all subcommands with the `-v` switch.
    """
    import glob
    import torch
    import tempfile
    import uuid

    from threadpoolctl import threadpool_limits

    from kraken.lib.progress import KrakenProgressBar

    if ctx.meta['device'] != 'cpu':
        try:
            torch.ones(1, device=ctx.meta['device'])
        except AssertionError as e:
            if raise_on_error:
                raise
            logger.error(f'Device {ctx.meta["device"]} not available: {e.args[0]}.')
            ctx.exit(1)
    input_format_type = format_type if format_type != 'pdf' else 'image'
    raise_failed = raise_on_error
    if not template:
        output_mode = serializer
        output_template = serializer
    else:
        output_mode = 'template'
        output_template = template

    message(f'Loading model {model}\t', nl=False)
    try:
        lm = torch.load(location, map_location=torch.device(ctx.meta['device']))
        model_weights = lm['state_dict']
        class_mapping = lm['BaselineDataModule']['class_mapping']
        config = SegformerConfig.from_dict(lm['model_config'])
        net = SegformerForSemanticSegmentation(config)
        for key in list(model_weights):
            model_weights[key.replace("net.", "")] = model_weights.pop(key)
        net.load_state_dict(model_weights)
    except Exception:
        if ctx.meta['raise_failed']:
            raise
        message('\u2717', fg='red')
        ctx.exit(1)

    message('\u2713', fg='green')

    ctx.meta['steps'].append({'category': 'processing',
                              'description': 'Baseline and region segmentation',
                              'settings': {'model': os.path.basename(model),
                                           'text_direction': text_direction}})

    input = list(input)
    # expand batch inputs
    if batch_input and suffix:
        for batch_expr in batch_input:
            for in_file in glob.glob(batch_expr, recursive=True):
                input.append((in_file, '{}{}'.format(os.path.splitext(in_file)[0], suffix)))

    # parse pdfs
    if format_type == 'pdf':
        import pyvips

        if not batch_input:
            logger.warning('PDF inputs not added with batch option. Manual output filename will be ignored and `-o` utilized.')
        new_input = []
        num_pages = 0
        for (fpath, _) in input:
            doc = pyvips.Image.new_from_file(fpath, dpi=300, n=-1, access="sequential")
            if 'n-pages' in doc.get_fields():
                num_pages += doc.get('n-pages')

        with KrakenProgressBar() as progress:
            pdf_parse_task = progress.add_task('Extracting PDF pages', total=num_pages, visible=True if not ctx.meta['verbose'] else False)
            for (fpath, _) in input:
                try:
                    doc = pyvips.Image.new_from_file(fpath, dpi=300, n=-1, access="sequential")
                    if 'n-pages' not in doc.get_fields():
                        logger.warning('{fpath} does not contain pages. Skipping.')
                        continue
                    n_pages = doc.get('n-pages')

                    dest_dict = {'idx': -1, 'src': fpath, 'uuid': None}
                    for i in range(0, n_pages):
                        dest_dict['idx'] += 1
                        dest_dict['uuid'] = str(uuid.uuid4())
                        fd, filename = tempfile.mkstemp(suffix='.png')
                        os.close(fd)
                        doc = pyvips.Image.new_from_file(fpath, dpi=300, page=i, access="sequential")
                        logger.info(f'Saving temporary image {fpath}:{dest_dict["idx"]} to {filename}')
                        doc.write_to_file(filename)
                        new_input.append((filename, pdf_format.format(**dest_dict) + suffix))
                        progress.update(pdf_parse_task, advance=1)
                except pyvips.error.Error:
                    num_pages -= n_pages
                    progress.update(pdf_parse_task, total=num_pages)
                    logger.warning(f'{fpath} is not a PDF file. Skipping.')
        input = new_input
        ctx.meta['steps'].insert(0, {'category': 'preprocessing', 'description': 'PDF image extraction', 'settings': {}})

    for io_pair in input:
        try:
            with threadpool_limits(limits=threads):
                _segment(input=input, output=output)
        except Exception as e:
            logger.error(f'Failed processing {io_pair[0]}: {str(e)}')
            if raise_failed:
                raise

    def _segment(input, output):
        if input_format_type != 'image':
            input = get_input_parser(input_format_type)(input).imagename

        try:
            im = Image.open(input)
        except IOError as e:
            raise click.BadParameter(str(e))
        message('Segmenting\t', nl=False)
        try:
            res = blla.segment(im, text_direction, mask=mask, model=model, device=device,
                               raise_on_error=raise_failed, autocast=precision)
        except Exception:
            if raise_failed:
                raise
            message('\u2717', fg='red')
            ctx.exit(1)

        if output_mode != 'native':
            with click.open_file(output, 'w', encoding='utf-8') as fp:
                logger.info('Serializing as {} into {}'.format(output_mode, output))
                from kraken import serialization
                fp.write(serialization.serialize_segmentation(res,
                                                              image_name=input,
                                                              image_size=im.size,
                                                              template=output_template,
                                                              template_source='custom' if output_mode == 'template' else 'native',
                                                              processing_steps=steps))
        else:
            with click.open_file(output, 'w') as fp:
                json.dump(dataclasses.asdict(res), fp)
        message('\u2713', fg='green')


    return partial(segmenter, boxes, model, text_direction, scale, maxcolseps,
                   black_colseps, remove_hlines, pad, mask, ctx.meta['device'])