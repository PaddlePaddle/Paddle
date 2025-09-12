# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

# The file has been adapted from DeepSeek DeepEP project
# Copyright (c) 2025 DeepSeek
# Licensed under the MIT License - https://github.com/deepseek-ai/DeepEP/blob/main/LICENSE

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import paddle
    from paddle.base.core import EventHandle

import paddle


class EventOverlap:
    """
    A wrapper class to manage CUDA events, also for better overlapping convenience.

    Attributes:
        event: the CUDA event captured.
        extra_tensors: an easier way to simulate tensor `record_stream`, may be useful with CUDA graph.
    """

    def __init__(
        self,
        event: EventHandle | None = None,
        extra_tensors: tuple[paddle.Tensor] | None = None,
    ) -> None:
        """
        Initialize the class.

        Arguments:
            event: the CUDA event captured.
            extra_tensors: an easier way to simulate tensor `record_stream`, may be useful with CUDA graph.
        """
        self.event = event

        # NOTES: we use extra tensors to achieve stream recording, otherwise,
        # stream recording will be incompatible with CUDA graph.
        self.extra_tensors = extra_tensors

    def current_stream_wait(self) -> None:
        """
        The current stream waits for the event to be finished.
        """
        assert self.event is not None
        self.event.current_stream_wait()

    def calc_stream_wait(self, group_idx) -> None:
        self.event.calc_stream_wait(group_idx)

    def comm_stream_wait(self, group_idx) -> None:
        self.event.comm_stream_wait(group_idx)

    def __enter__(self) -> Any:
        """
        Utility for overlapping and Python `with` syntax.

        You can overlap the kernels on the current stream with the following example:
        ```python
        event_overlap = event_after_all_to_all_kernels()
        with event_overlap():
            do_something_on_current_stream()
        # After exiting the `with` scope, the current stream with wait the event to be finished.
        ```
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Utility for overlapping and Python `with` syntax.

        Please follow the example in the `__enter__` function.
        """
        if self.event is not None:
            self.event.current_stream_wait()


def get_event_from_calc_stream(group_id: int) -> EventOverlap:
    return EventOverlap(
        event=paddle.base.core.get_event_handle_from_calc_stream(group_id)
    )


def get_event_from_comm_stream(group_id: int) -> EventOverlap:
    return EventOverlap(
        event=paddle.base.core.get_event_handle_from_comm_stream(group_id)
    )


def get_event_from_custom_stream(stream) -> EventOverlap:
    return EventOverlap(
        event=paddle.base.core.get_event_handle_from_custom_stream(stream)
    )
