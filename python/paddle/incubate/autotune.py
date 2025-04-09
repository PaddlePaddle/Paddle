# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
import warnings
from typing import TYPE_CHECKING, TypedDict

import paddle
from paddle.base import core

if TYPE_CHECKING:
    from typing_extensions import NotRequired

    class _Kernel(TypedDict):
        enable: bool
        tuning_range: list[int] | tuple[int, int]

    class _Layout(TypedDict):
        enable: bool

    class _Dataloader(TypedDict):
        enable: bool
        tuning_steps: int

    class _ConfigKernel(TypedDict):
        kernel: NotRequired[_Kernel]
        layout: NotRequired[_Layout]
        dataloader: NotRequired[_Dataloader]


__all__ = ['set_config']


def set_config(config: _ConfigKernel | str | None = None) -> None:
    r"""
    Set the configuration for kernel, layout and dataloader auto-tuning.

    1. kernel: When it is enabled, exhaustive search method will be used to select
    and cache the best algorithm for the operator in the tuning iteration. Tuning
    parameters are as follows:

    - enable(bool): Whether to enable kernel tuning.
    - tuning_range(list): Start and end iteration for auto-tuning. Default: [1, 10].

    2. layout: When it is enabled, the best data layout such as NCHW or NHWC will be
    determined based on the device and data type. When the origin layout setting is
    not best, layout transformation will be automatically performed to improve model
    performance. Layout auto-tuning only supports dygraph mode currently. Tuning
    parameters are as follows:

    - enable(bool): Whether to enable layout tuning.

    3. dataloader: When it is enabled, the best num_workers will be selected to replace
    the origin dataloader setting. Tuning parameters are as follows:

    - enable(bool): Whether to enable dataloader tuning.

    Args:
        config (dict|str|None, optional): Configuration for auto-tuning. If it is a
            dictionary, the key is the tuning type, and the value is a dictionary
            of the corresponding tuning parameters. If it is a string, the path of
            a json file will be specified and the tuning configuration will be set
            by the json file. Default: None, auto-tuning for kernel, layout and
            dataloader will be enabled.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import json

            >>> # config is a dict.
            >>> config = {
            ...     "kernel": {
            ...         "enable": True,
            ...         "tuning_range": [1, 5],
            ...     },
            ...     "layout": {
            ...         "enable": True,
            ...     },
            ...     "dataloader": {
            ...         "enable": True,
            ...     }
            >>> }
            >>> paddle.incubate.autotune.set_config(config) # type: ignore[arg-type]

            >>> # config is the path of json file.
            >>> config_json = json.dumps(config)
            >>> with open('config.json', 'w') as json_file:
            ...     json_file.write(config_json)
            >>> paddle.incubate.autotune.set_config('config.json')

    """
    if config is None:
        core.enable_autotune()
        core.enable_layout_autotune()
        paddle.io.reader.set_autotune_config(use_autotune=True)
        return

    config_dict = {}
    if isinstance(config, dict):
        config_dict = config
    elif isinstance(config, str):
        try:
            with open(config, 'r') as filehandle:
                config_dict = json.load(filehandle)
        except Exception as e:
            print(f'Load config error: {e}')
            warnings.warn("Use default configuration for auto-tuning.")

    if "kernel" in config_dict:
        kernel_config = config_dict["kernel"]
        if "enable" in kernel_config:
            if isinstance(kernel_config['enable'], bool):
                if kernel_config['enable']:
                    core.enable_autotune()
                else:
                    core.disable_autotune()
            else:
                warnings.warn(
                    "The auto-tuning configuration of the kernel is incorrect."
                    "The `enable` should be bool. Use default parameter instead."
                )
        if "tuning_range" in kernel_config:
            if isinstance(kernel_config['tuning_range'], list):
                tuning_range = kernel_config['tuning_range']
                assert len(tuning_range) == 2
                core.set_autotune_range(tuning_range[0], tuning_range[1])
            else:
                warnings.warn(
                    "The auto-tuning configuration of the kernel is incorrect."
                    "The `tuning_range` should be list. Use default parameter instead."
                )
    if "layout" in config_dict:
        layout_config = config_dict["layout"]
        if "enable" in layout_config:
            if isinstance(layout_config['enable'], bool):
                if layout_config['enable']:
                    core.enable_layout_autotune()
                else:
                    core.disable_layout_autotune()
            else:
                warnings.warn(
                    "The auto-tuning configuration of the layout is incorrect."
                    "The `enable` should be bool. Use default parameter instead."
                )
    if "dataloader" in config_dict:
        dataloader_config = config_dict["dataloader"]
        use_autotune = False
        if "enable" in dataloader_config:
            if isinstance(dataloader_config['enable'], bool):
                use_autotune = dataloader_config['enable']
            else:
                warnings.warn(
                    "The auto-tuning configuration of the dataloader is incorrect."
                    "The `enable` should be bool. Use default parameter instead."
                )
        if "tuning_steps" in dataloader_config:
            if isinstance(dataloader_config['tuning_steps'], int):
                paddle.io.reader.set_autotune_config(
                    use_autotune, dataloader_config['tuning_steps']
                )
            else:
                warnings.warn(
                    "The auto-tuning configuration of the dataloader is incorrect."
                    "The `tuning_steps` should be int. Use default parameter instead."
                )
                paddle.io.reader.set_autotune_config(use_autotune)
