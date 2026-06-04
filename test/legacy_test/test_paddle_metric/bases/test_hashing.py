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

import pytest
from unittests._helpers.testers import DummyListMetric, DummyMetric


@pytest.mark.parametrize("metric_cls", [DummyMetric, DummyListMetric])
def test_metric_hashing(metric_cls):
    """Tests that hashes are different.

    See the Metric's hash function for details on why this is required.

    """
    instance_1 = metric_cls()
    instance_2 = metric_cls()
    assert hash(instance_1) != hash(instance_2)
    assert id(instance_1) != id(instance_2)
