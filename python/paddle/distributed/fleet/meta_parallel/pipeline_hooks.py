# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import annotations

from collections import defaultdict
from enum import Enum
from typing import Callable


# Enum for specifying the pipeline parallel micro-step locations.
class PipelineParallelMicroStepLocations(Enum):
    FORWARD_BEGIN = 'forward_begin'
    FORWARD_END = 'forward_end'
    BACKWARD_BEGIN = 'backward_begin'
    BACKWARD_END = 'backward_end'


# A callback class for managing hooks at different stages of a pipeline parallel process.
class PipelineParallelMicroStepCallback:
    def __init__(self):
        # Initializes a dictionary to store hooks for each micro-step location in the pipeline.
        self.hooks: dict[PipelineParallelMicroStepLocations, list[Callable]] = {
            PipelineParallelMicroStepLocations.FORWARD_BEGIN: [],
            PipelineParallelMicroStepLocations.FORWARD_END: [],
            PipelineParallelMicroStepLocations.BACKWARD_BEGIN: [],
            PipelineParallelMicroStepLocations.BACKWARD_END: [],
        }

    def register_hook(
        self, location: PipelineParallelMicroStepLocations, hook: Callable
    ):
        """
        Registers a hook function to be called at a specified pipeline parallel micro-step location.

        Args:
            location (PipelineParallelMicroStepLocations): The micro-step location where the hook should be registered.
            hook (Callable): The hook function to be registered. The function should accept the following optional keyword arguments:
                - input_tensor (paddle.Tensor): The input tensor to the current micro-step.
                - output_tensor (paddle.Tensor): The output tensor from the current micro-step.
                - input_tensor_grad (paddle.Tensor): The gradient of the input tensor.
                - output_tensor_grad (paddle.Tensor): The gradient of the output tensor.
                - step_id (paddle.Tensor): An identifier for the current step in the pipeline.

        Raises:
            AssertionError: If the specified location is not a valid micro-step location.
        """
        assert (
            location in self.hooks
        ), f"Invalid location '{location}'. Valid locations are 'forward_begin', 'forward_end', 'backward_begin', or 'backward_end'."
        self.hooks[location].append(hook)

    def on_location(
        self, location: PipelineParallelMicroStepLocations, **kwargs
    ):
        """
        Triggers all registered hooks at a specified pipeline parallel micro-step location.

        Args:
            location (PipelineParallelMicroStepLocations): The micro-step location where the hooks should be triggered.
            kwargs: Additional keyword arguments to be passed to the hook functions.

        Raises:
            AssertionError: If the specified location is not a valid micro-step location.
        """
        assert (
            location in self.hooks
        ), f"Invalid location '{location}'. Valid locations are 'forward_begin', 'forward_end', 'backward_begin', or 'backward_end'."
        for hook in self.hooks[location]:
            hook(**kwargs)


class BubbleHook:
    def __init__(self):
        # self.hooks: dict[int, list[Callable]] = {}
        self.hooks: dict[int, list[Callable]] = defaultdict(list)

    def set_bubble_times(self, bubble_times):
        self.bubble_times = bubble_times

    def register_hook(self, location: int, hook: Callable):
        # assert (
        #    location < self.bubble_times
        # ), f"register hook location[{location}] should be less than or equal bubble_times[{self.bubble_times}]"
        self.hooks[location].append(hook)

    def on_location(self, location: int, **kwargs):
        print(f"on_location: {kwargs}")
        if location not in self.hooks:
            return

        for hook in self.hooks[location]:
            hook(**kwargs)
