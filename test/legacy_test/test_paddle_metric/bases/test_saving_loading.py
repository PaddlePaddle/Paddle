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

import paddle
from paddle.metric.classification import MulticlassAccuracy


@pytest.mark.parametrize("persistent", [True, False])
@pytest.mark.parametrize("in_device", ["cpu", "cuda"])
@pytest.mark.parametrize("out_device", ["cpu", "cuda"])
def test_saving_loading(persistent, in_device, out_device):
    """Test that saving and loading works as expected."""
    if (
        in_device == "cuda" or out_device == "cuda"
    ) and not paddle.cuda.is_available():
        pytest.skip("Test requires cuda, but GPU not available.")
    metric1 = MulticlassAccuracy(num_classes=5).to(in_device)
    metric1.persistent(persistent)
    metric1.update(
        paddle.randint(low=0, high=5, shape=(100,)).to(in_device),
        paddle.randint(low=0, high=5, shape=(100,)).to(in_device),
    )
    paddle.save(obj=metric1.state_dict(), path="metric.pth")
    metric2 = MulticlassAccuracy(num_classes=5).to(out_device)
    metric2.load_state_dict(paddle.load(path="metric.pth"))
    metric_state1 = metric1.metric_state
    metric_state2 = metric2.metric_state
    for k, v in metric_state1.items():
        v2 = metric_state2[k]
        if in_device == out_device:
            if persistent:
                assert paddle.allclose(x=v, y=v2).item()
            else:
                assert not paddle.allclose(x=v, y=v2).item()
        elif persistent:
            assert paddle.allclose(x=v, y=v2.to(v.place)).item()
        else:
            assert not paddle.allclose(x=v, y=v2.to(v.place)).item()
