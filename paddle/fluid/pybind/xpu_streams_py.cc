// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/pybind/xpu_streams_py.h"

#include <string>
#include <vector>

#include "paddle/fluid/platform/device_event_base.h"
#include "paddle/fluid/platform/event.h"
#if defined(PADDLE_WITH_XPU)
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#endif

namespace py = pybind11;

namespace paddle {
namespace pybind {
void BindXpuStream(py::module *m_ptr) {
  auto &m = *m_ptr;

  // Bind Methods
  m.def("_xpu_device_synchronize", [](int device_id) {
#if defined(PADDLE_WITH_XPU)
    if (device_id == -1) {
      device_id = paddle::platform::GetXPUCurrentDeviceId();
    }
    int curr_device_id = paddle::platform::GetXPUCurrentDeviceId();
    paddle::platform::SetXPUDeviceId(device_id);
    PADDLE_ENFORCE_XPU_SUCCESS(xpu_wait());
    paddle::platform::SetXPUDeviceId(curr_device_id);
#else
    PADDLE_THROW(platform::errors::Unavailable(
        "Paddle is not compiled with XPU. Cannot visit device synchronize."));
#endif
  });
}

}  // namespace pybind
}  // namespace paddle
