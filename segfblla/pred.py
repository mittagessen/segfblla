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
import torch
import warnings
from os import PathLike
from functools import partial
from pathlib import Path
from typing import IO, Any, Callable, Dict, List, Union, Tuple

from transformers import SegformerConfig, SegformerForSemanticSegmentation

from kraken.blla import vec_regions, vec_lines
from kraken.lib.segmentation import polygonal_reading_order

def load_model_checkpoint(filename: PathLike, device: torch.device) -> torch.nn.Module:
    """
    Instantiates a pure torch nn.Module from a lightning checkpoint and returns
    the class mapping.
    """
    lm = torch.load(location, map_location=device)
    model_weights = lm['state_dict']
    class_mapping = lm['BaselineDataModule']['class_mapping']
    config = SegformerConfig.from_dict(lm['model_config'])
    net = SegformerForSemanticSegmentation(config)
    for key in list(model_weights):
        model_weights[key.replace("net.", "")] = model_weights.pop(key)
    net.load_state_dict(model_weights)
    net.class_mapping = lm['BaselineDataModule']['class_mapping']
    net.topline = lm['BaselineDataModule']['hyper_parameters']['topline']
    return net


def compute_segmentation_map(im: PIL.Image.Image,
                             model: nn.Module = None,
                             device: torch.device = torch.device('cpu'),
                             autocast: bool = True) -> Dict[str, Any]:
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

    tensor_im = transforms(im)
    with torch.autocast(device_type=device.split(":")[0], enabled=autocast):
        with torch.no_grad():
            logger.debug('Running network forward pass')
            o, _ = model.nn(tensor_im.to(device))
    logger.debug('Reassembling patches')

    logger.debug('Upsampling network output')
    o = F.interpolate(o, size=im.shape)
    o = o.squeeze().cpu().float().numpy()
    scale = np.divide(im.size, o.shape[:0:-1])

    return {'heatmap': o,
            'cls_map': model.class_mapping,
            'bounding_regions': None,
            'scale': scale,
            'scal_im': scal_im}


def segment(im: PIL.Image.Image,
            text_direction: Literal['horizontal-lr', 'horizontal-rl', 'vertical-lr', 'vertical-rl'] = 'horizontal-lr',
            reading_order_fn: Callable = polygonal_reading_order,
            model: nn.Module = None,
            device: torch.device = torch.device('cpu'),
            raise_on_error: bool = False,
            autocast: bool = True) -> Segmentation:
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
    order = None

    rets = compute_segmentation_map(im, mask, model, device, autocast=autocast)
    regions = vec_regions(**rets)

    # flatten regions for line ordering
    line_regs = []
    for cls, regs in regions.items():
        line_regs.extend(regs)

    # convert back to net scale
    line_regs = scale_regions([x.boundary for x in line_regs], 1/rets['scale'])

    lines = vec_lines(**rets,
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

