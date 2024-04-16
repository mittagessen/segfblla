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
Default hyperparameters
"""

SEGMENTATION_HYPER_PARAMS = {'line_width': 8,
                             'batch_size': 16,
                             'freq': 1.0,
                             'quit': 'fixed',
                             'epochs': 500,
                             'min_epochs': 0,
                             'lag': 10,
                             'min_delta': None,
                             'optimizer': 'AdamW',
                             'lrate': 5e-6,
                             'momentum': 0.9,
                             'weight_decay': 1e-5,
                             'schedule': 'cosine',
                             'completed_epochs': 0,
                             'augment': True,
                             # lr scheduler params
                             # step/exp decay
                             'step_size': 10,
                             'gamma': 0.1,
                             # reduce on plateau
                             'rop_factor': 0.1,
                             'rop_patience': 5,
                             # cosine
                             'cos_t_max': 50,
                             'cos_min_lr': 1e-8,
                             'warmup': 1500,
                             'patch_size': (512, 512),
                             'loss': 'bce',
                             }
