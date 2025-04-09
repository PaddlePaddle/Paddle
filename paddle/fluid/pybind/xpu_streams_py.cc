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

#include "paddle/phi/api/profiler/event.h"
#include "paddle/phi/core/platform/device_event_base.h"

#if defined(PADDLE_WITH_XPU)
#include <cuda.h>
#include <cuda_runtime.h>
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "xpu/runtime.h"
#include "xpu/runtime_ex.h"
#endif

namespace py = pybind11;

namespace paddle {
namespace platform {
#ifdef PADDLE_WITH_XPU
XPUStream get_current_stream(int device_id) {
  if (device_id == -1) {
    device_id = phi::backends::xpu::GetXPUCurrentDeviceId();
  }
  auto place = phi::XPUPlace(device_id);
  auto *dev_ctx = static_cast<phi::XPUContext *>(
      phi::DeviceContextPool::Instance().Get(place));
  dev_ctx->Wait();
  return dev_ctx->stream();
}

#endif
}  // namespace platform
namespace pybind {
void BindXpuStream(py::module *m_ptr) {
  auto &m = *m_ptr;

  // Bind Methods
  m.def("_xpu_device_synchronize", [](int device_id) {
#ifdef PADDLE_WITH_XPU
    if (device_id == -1) {
      device_id = paddle::platform::GetXPUCurrentDeviceId();
    }
    int curr_device_id = paddle::platform::GetXPUCurrentDeviceId();
    paddle::platform::SetXPUDeviceId(device_id);
    auto place = phi::XPUPlace(device_id);
    auto *dev_ctx = static_cast<phi::XPUContext *>(
        phi::DeviceContextPool::Instance().Get(place));
    dev_ctx->Wait();
    paddle::platform::SetXPUDeviceId(curr_device_id);
#else
    PADDLE_THROW(common::errors::Unavailable(
        "Paddle is not compiled with XPU. Cannot visit device synchronize."));
#endif
  });
  m.def(
      "_get_current_stream",
      [](int device_id) {
#ifdef PADDLE_WITH_XPU
        if (device_id == -1) {
          device_id = paddle::platform::GetXPUCurrentDeviceId();
        }
        paddle::platform::SetXPUDeviceId(device_id);
        return platform::get_current_stream(device_id);
#else
        PADDLE_THROW(
            common::errors::Unavailable("Paddle is not compiled with CUDA. "
                                        "Cannot visit device synchronize."));
#endif
      },
      py::return_value_policy::reference);
  m.def("_device_synchronize", [](int device_id) {
#ifdef PADDLE_WITH_XPU
    if (device_id == -1) {
      device_id = paddle::platform::GetXPUCurrentDeviceId();
    }

    int curr_device_id = paddle::platform::GetXPUCurrentDeviceId();
    paddle::platform::SetXPUDeviceId(device_id);
    PADDLE_ENFORCE_XPU_SUCCESS(cudaDeviceSynchronize());
    paddle::platform::SetXPUDeviceId(curr_device_id);
#else
    PADDLE_THROW(common::errors::Unavailable(
        "Paddle is not compiled with CUDA. Cannot visit device synchronize."));
#endif
  });

#ifdef PADDLE_WITH_XPU
  py::class_<XPUStream>(m, "XPUStream", R"DOC(
      The handle of the CUDA stream.

      Parameters:
          device(paddle.CUDAPlace()|int|None, optional): The device which wanted to allocate the stream.
              If device is None or negative integer, device will be the current device.
              If device is positive integer, it must less than the device count. Default: None.
          priority(int|None, optional): The priority of stream. The priority can be 1(high) or 2(normal).
              If priority is None, the priority is 2(normal). Default: None.

      Examples:
          .. code-block:: python

              >>> # doctest: +REQUIRES(env:GPU)
              >>> import paddle
              >>> s1 = paddle.device.cuda.Stream(paddle.CUDAPlace(0), 1)
              >>> s2 = paddle.device.cuda.Stream(0, 1)
              >>> s3 = paddle.device.cuda.Stream()

      )DOC")
      .def(
          "synchronize",
          [](XPUStream &self) { xpu_wait(self); },
          R"DOC(
          Waits for stream tasks to complete.

          Examples:
              .. code-block:: python

                  >>> # doctest: +REQUIRES(env:GPU)
                  >>> import paddle
                  >>> s = paddle.device.cuda.Stream(paddle.CUDAPlace(0), 1)
                  >>> s.synchronize()

          )DOC");
#endif
}
}  // namespace pybind
}  // namespace paddle
