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
import PIL
import uuid
import torch
import logging
import numpy as np
import shapely.geometry as geom
import torch.nn.functional as F

from typing import Any, Callable, Dict, Literal, TYPE_CHECKING
from torchvision.transforms import v2

from transformers import SegformerConfig, SegformerForSemanticSegmentation

from segfblla.tiles import ImageSlicer

from kraken.blla import vec_regions, vec_lines
from kraken.containers import Segmentation, BaselineLine
from kraken.lib.segmentation import polygonal_reading_order, is_in_region

if TYPE_CHECKING:
    from os import PathLike
    from torch import nn

logger = logging.getLogger(__name__)


def load_model_checkpoint(filename: 'PathLike', device: torch.device) -> 'nn.Module':
    """
    Instantiates a pure torch nn.Module from a lightning checkpoint and returns
    the class mapping.
    """
    lm = torch.load(filename, map_location=device)
    model_weights = lm['state_dict']
    config = SegformerConfig.from_dict(lm['model_config'])
    net = SegformerForSemanticSegmentation(config)
    for key in list(model_weights):
        model_weights[key.replace("net.", "")] = model_weights.pop(key)
    net.load_state_dict(model_weights)
    net.class_mapping = lm['BaselineDataModule']['class_mapping']
    net.topline = lm['BaselineDataModule'].get('topline', False)
    net.patch_size = lm['BaselineDataModule'].get('patch_size', (512, 512))
    return net


def compute_segmentation_map(im: PIL.Image.Image,
                             model: 'nn.Module' = None,
                             device: torch.device = torch.device('cpu'),
                             batch_size: int = 1,
                             autocast: bool = True,
                             callback: Callable[[int, int], Any] = None) -> Dict[str, Any]:
    """
    Args:
        im: Input image
        model: A TorchVGSLModel containing a segmentation model.
        device: The target device to run the neural network on.
        autocast: Runs the model with automatic mixed precision

    Returns:
        A dictionary containing the heatmaps ('heatmap', torch.Tensor), class
        map ('cls_map', Dict[str, Dict[str, int]]), the bounding regions for
        polygonization purposes ('bounding_regions', List[str]), the scale
        between the input image and the network output ('scale', float), and
        the scaled input image to the network ('scal_im', PIL.Image.Image).
    """
    model.eval()
    model.to(device)

    tiler = ImageSlicer(im.size[::-1],
                        tile_size=model.patch_size,
                        overlap=(32, 32),
                        batch_size=batch_size)

    callback(tiler.num_batches, 0)

    transforms = v2.Compose([v2.PILToTensor(),
                             v2.ConvertDtype(torch.float32),
                             v2.Normalize(mean=(0.485, 0.456, 0.406),
                                          std=(0.229, 0.224, 0.225)),
                            ]
                           )

    tensor_im = transforms(im)

    with torch.autocast(device_type=str(device).split(":")[0], enabled=autocast):
        with torch.no_grad():
            tiles = []
            for idx, tile in enumerate(tiler.split(tensor_im)):
                logger.debug('Running network forward pass')
                callback(tiler.num_batches, 1)
                tiles.append(torch.sigmoid(model(tile.to(device)).logits))
            tiles = torch.cat(tiles)
    logger.debug('Upsampling network output')
    tiles = F.interpolate(tiles, tiler.tile_size, mode='bicubic')
    logger.debug('Reassembling patches')
    o = tiler.merge(tiles)
    o = o.squeeze().cpu().float().numpy()

    return {'heatmap': o,
            'cls_map': model.class_mapping,
            'bounding_regions': None,
            'scal_im': np.array(im)}


def segment(im: PIL.Image.Image,
            text_direction: Literal['horizontal-lr', 'horizontal-rl', 'vertical-lr', 'vertical-rl'] = 'horizontal-lr',
            reading_order_fn: Callable = polygonal_reading_order,
            model: 'nn.Module' = None,
            device: torch.device = torch.device('cpu'),
            raise_on_error: bool = False,
            autocast: bool = True,
            batch_size: int = 4,
            callback: Callable[[int, int], Any] = None) -> Segmentation:
    r"""
    Segments a page into text lines using the baseline segmenter.

    Segments a page into text lines and returns the polyline formed by each
    baseline and their estimated environment.

    Args:
        im: Input image. The mode can generally be anything but it is possible
            to supply a binarized-input-only model which requires accordingly
            treated images.
        text_direction: Passed-through value for serialization.serialize.
        reading_order_fn: Function to determine the reading order.  Has to
                          accept a list of tuples (baselines, polygon) and a
                          text direction (`lr` or `rl`).
        model: One or more TorchVGSLModel containing a segmentation model. If
               none is given a default model will be loaded.
        device: The target device to run the neural network on.
        raise_on_error: Raises error instead of logging them when they are
                        not-blocking
        precision: Runs the model with automatic mixed precision

    Returns:
        A :class:`kraken.containers.Segmentation` class containing reading
        order sorted baselines (polylines) and their respective polygonal
        boundaries as :class:`kraken.containers.BaselineLine` records. The last
        and first point of each boundary polygon are connected.
    """
    loc = {None: 'center',
           True: 'top',
           False: 'bottom'}[model.topline]
    logger.debug(f'Baseline location: {loc}')

    lines = []

    def _callback(_total, advance):
        global total
        total = _total
        if callback:
            callback(_total + 3, advance)

    rets = compute_segmentation_map(im,
                                    model,
                                    device,
                                    batch_size=batch_size,
                                    autocast=autocast,
                                    callback=_callback)
    callback(total+3, 1)
    regions = vec_regions(**rets, scale=1.0)

    # flatten regions for line ordering
    line_regs = []
    for cls, regs in regions.items():
        line_regs.extend([reg.boundary for reg in regs])

    callback(total+3, 1)
    lines = vec_lines(**rets,
                      scale=1.0,
                      regions=line_regs,
                      text_direction=text_direction,
                      topline=model.topline,
                      raise_on_error=raise_on_error)

    if len(rets['cls_map']['baselines']) > 1:
        script_detection = True
    else:
        script_detection = False

    # create objects and assign IDs
    blls = []
    _shp_regs = {}
    for reg_type, rgs in regions.items():
        for reg in rgs:
            _shp_regs[reg.id] = geom.Polygon(reg.boundary)

    callback(total+3, 1)
    # reorder lines
    logger.debug(f'Reordering baselines with main RO function {reading_order_fn}.')
    basic_lo = reading_order_fn(lines=lines, regions=_shp_regs.values(), text_direction=text_direction[-2:])
    lines = [lines[idx] for idx in basic_lo]

    for line in lines:
        line_regs = []
        for reg_id, reg in _shp_regs.items():
            line_ls = geom.LineString(line['baseline'])
            if is_in_region(line_ls, reg):
                line_regs.append(reg_id)
        blls.append(BaselineLine(id=str(uuid.uuid4()), baseline=line['baseline'], boundary=line['boundary'], tags=line['tags'], regions=line_regs))

    return Segmentation(text_direction=text_direction,
                        imagename=getattr(im, 'filename', None),
                        type='baselines',
                        lines=blls,
                        regions=regions,
                        script_detection=script_detection,
                        line_orders=[])
