# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import warnings
from typing import TYPE_CHECKING, Any

from paddle.io import DataLoader as PaddleDataLoader

from ._utils.collate import (
    default_collate as default_collate,
)
from ._utils.worker import (
    get_worker_info as get_worker_info,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from paddle.io.dataloader import BatchSampler
    from paddle.io.dataloader.dataset import Dataset
    from paddle.io.reader import _CollateFn


class DataLoader(PaddleDataLoader):
    def __init__(
        self,
        dataset: Dataset[Any],
        batch_size: int | None = 1,
        shuffle: bool = False,
        sampler: BatchSampler | None = None,
        batch_sampler: BatchSampler | None = None,
        num_workers: int = 0,
        collate_fn: _CollateFn | None = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Callable[[int], None] | None = None,
        multiprocessing_context=None,
        generator=None,
        *,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
        pin_memory_device: str = "",
        in_order: bool = True,
    ) -> None:
        if (
            pin_memory is True
            or multiprocessing_context is not None
            or generator is not None
            or prefetch_factor is not None
            or len(pin_memory_device) > 0
            or in_order is False
        ):
            warnings.warn(
                "pin_memory, multiprocessing_context, generator, prefetch_factor, pin_memory_device, in_order are currently not supported in DataLoader and will be ignored."
            )

        if sampler is not None:
            if batch_sampler is not None:
                raise ValueError(
                    "Cannot specify both sampler and batch_sampler"
                )
            batch_sampler = sampler

        super().__init__(
            dataset=dataset,
            batch_sampler=batch_sampler,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=collate_fn,
            num_workers=num_workers,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            persistent_workers=persistent_workers,
        )
