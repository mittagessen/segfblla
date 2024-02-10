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
                             'freq': 1.0,
                             'quit': 'fixed',
                             'epochs': 50,
                             'min_epochs': 0,
                             'lag': 10,
                             'min_delta': None,
                             'optimizer': 'AdamW',
                             'lrate': 0.00006,
                             'momentum': 0.9,
                             'weight_decay': 1e-5,
                             'schedule': 'constant',
                             'completed_epochs': 0,
                             'augment': False,
                             # lr scheduler params
                             # step/exp decay
                             'step_size': 10,
                             'gamma': 0.1,
                             # reduce on plateau
                             'rop_factor': 0.1,
                             'rop_patience': 5,
                             # cosine
                             'cos_t_max': 50,
                             'warmup': 0,
                             }
