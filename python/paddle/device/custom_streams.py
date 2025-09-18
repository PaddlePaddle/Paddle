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
from __future__ import annotations

from paddle.base.core import (
    CustomDeviceEvent as Event,
    CustomDeviceStream as Stream,
    CustomPlace,
)


def create_stream(
    device_id: CustomPlace | int | None = None,
    priority: int = 2,
    device_type: str | None = None,  # Ignored for compatibility
    blocking: bool = False,  # Ignored for compatibility
):
    """
    Factory Function, used to create Custom Stream
    """
    return Stream(
        device_type,
        device_id,
        priority,
        blocking=blocking,
    )


def create_event(
    enable_timing: bool = False,
    blocking: bool = False,
    interprocess: bool = False,
    device_type: str | None = None,
    device_id: int = 0,
):
    """
    Factory Function, used to create Custom Event
    """
    return Event(
        device_type,
        device_id,
        enable_timing,
        blocking,
        interprocess,
    )
