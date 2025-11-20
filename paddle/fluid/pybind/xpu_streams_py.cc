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
phi::XPUStreamHandle *get_current_stream(int device_id) {
  auto place = phi::XPUPlace(device_id);
  auto *dev_ctx = static_cast<phi::XPUContext *>(
      phi::DeviceContextPool::Instance().Get(place));
  dev_ctx->Wait();
  return dev_ctx->get_current_stream_handle();
}

phi::XPUStreamHandle *set_current_stream(int idx) {
  int device_id = phi::backends::xpu::GetXPUCurrentDeviceId();
  auto original_stream = get_current_stream(device_id);
  auto place = phi::XPUPlace(device_id);
  auto *dev_ctx = static_cast<phi::XPUContext *>(
      phi::DeviceContextPool::Instance().Get(place));
  dev_ctx->SetCurrentStream(idx);
  return original_stream;
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
      "_xpu_get_current_stream",
      [](int device_id) {
#ifdef PADDLE_WITH_XPU
        if (device_id == -1) {
          device_id = paddle::platform::GetXPUCurrentDeviceId();
        }
        paddle::platform::SetXPUDeviceId(device_id);
        return platform::get_current_stream(device_id);
#else
        PADDLE_THROW(
            common::errors::Unavailable("Paddle is not compiled with XPU. "
                                        "Cannot visit device synchronize."));
#endif
      },
      py::return_value_policy::reference);
  m.def(
      "_xpu_set_current_stream",
      [](int stream_id) {
#ifdef PADDLE_WITH_XPU
        return platform::set_current_stream(stream_id);
#else
        PADDLE_THROW(
            common::errors::Unavailable("Paddle is not compiled with XPU. "
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

  py::class_<phi::XPUStreamHandle>(m, "XPUStream", R"DOC(
      The handle of the XPU stream.

      Parameters:
          device(paddle.XPUPlace()|int|None, optional): The device which wanted to allocate the stream.
              If device is None or negative integer, device will be the current device.
              If device is positive integer, it must less than the device count. Default: None.

      Examples:
          .. code-block:: python

              >>> # doctest: +REQUIRES(env:XPU)
              >>> import paddle
              >>> s1 = paddle.device.xpu.Stream(paddle.XPUPlace(0))
              >>> s2 = paddle.device.xpu.Stream(0)
              >>> s3 = paddle.device.xpu.Stream()

      )DOC")
#ifdef PADDLE_WITH_XPU
      .def_property_readonly(
          "xpu_stream",
          [](phi::XPUStreamHandle &self) {
            return reinterpret_cast<std::uintptr_t>(self.raw_stream());
          })
      .def("wait_stream",
           [](phi::XPUStreamHandle &self, phi::XPUStreamHandle &other) {
             auto *dev_ctx = phi::get_xpu_context();
             dev_ctx->StreamWaitStreamInPool(self.id(), other.id());
           })
      .def("wait_event",
           [](phi::XPUStreamHandle &self, phi::XPUEventHandle &other) {
             self.wait_event(other.get_event());
           })
      .def("query",
           [](phi::XPUStreamHandle &self) {
             PADDLE_THROW(common::errors::Unavailable(
                 "Query function for XPUStream is not supported now"));
           })
      .def("record_event",
           [](phi::XPUStreamHandle &self, phi::XPUEventHandle *event) {
             if (event == nullptr) {
               event = new phi::XPUEventHandle();
             }
             self.record_event(event->get_event());
             return event;
           })
      .def(
          "synchronize",
          [](phi::XPUStreamHandle &self) { self.synchronize(); },
          R"DOC(
          Waits for stream tasks to complete.

          Examples:
              .. code-block:: python

                  >>> # doctest: +REQUIRES(env:XPU)
                  >>> import paddle
                  >>> s = paddle.device.xpu.Stream(paddle.XPUPlace(0), 1)
                  >>> s.synchronize()

          )DOC")
      .def_property_readonly(
          "place",
          [](phi::XPUStreamHandle &self) {
            return phi::XPUPlace(platform::GetXPUCurrentDeviceId());
          })
      .def_property_readonly(
          "idx", [](phi::XPUStreamHandle &self) { return self.id(); })
#endif

      .def("__init__",
           [](phi::XPUStreamHandle &self) {
#ifdef PADDLE_WITH_XPU
             new (&self) phi::XPUStreamHandle();
             self.Init();
#else
            PADDLE_THROW(common::errors::Unavailable(
                "Class XPUStream can only be initialized on the XPU "
                "platform."));
#endif
           })
      .def(
          "__init__",
          [](phi::XPUStreamHandle &self, phi::XPUPlace *place) {
#ifdef PADDLE_WITH_XPU
            if (place == nullptr) {
              int curr_device_id = platform::GetXPUCurrentDeviceId();
              auto place_tmp = phi::XPUPlace(curr_device_id);
              new (&self) phi::XPUStreamHandle(place_tmp);
            } else {
              new (&self) phi::XPUStreamHandle(*place);
            }
#else
            PADDLE_THROW(common::errors::Unavailable(
                "Class XPUStream can only be initialized on the XPU "
                "platform."));
#endif
          },
          py::arg("device") = nullptr)
      .def(
          "__init__",
          [](phi::XPUStreamHandle &self, int device) {
#ifdef PADDLE_WITH_XPU
            if (device < 0) {
              device = platform::GetXPUCurrentDeviceId();
            }
            auto place_tmp = phi::XPUPlace(device);
            new (&self) phi::XPUStreamHandle(place_tmp);
#else
            PADDLE_THROW(common::errors::Unavailable(
                "Class XPUStream can only be initialized on the XPU "
                "platform."));
#endif
          },
          py::arg("device") = -1);
  py::class_<phi::XPUEventHandle>(m, "XPUEvent", R"DOC(
      The handle of the XPU event.

      Examples:
          .. code-block:: python

              >>> # doctest: +REQUIRES(env:XPU)
              >>> import paddle
              >>> event = paddle.device.xpu.Event()

      )DOC")
#ifdef PADDLE_WITH_XPU
      .def(
          "record",
          [](phi::XPUEventHandle &self, phi::XPUStreamHandle *stream) {
            if (stream == nullptr) {
              auto *dev_ctx = phi::get_xpu_context();
              auto stream_handle = dev_ctx->get_current_stream_handle();
              self.record(stream_handle->raw_stream());
            } else {
              self.record(stream->raw_stream());
            }
          },
          py::arg("stream") = nullptr)
      .def("query", [](phi::XPUEventHandle &self) { return self.query(); })
      .def("elapsed_time",
           [](phi::XPUEventHandle &self) {
             PADDLE_THROW(common::errors::Unavailable(
                 "XPUEvent elapsed_time is not supported now"));
           })
      .def("synchronize", [](phi::XPUEventHandle &self) { self.synchronize(); })
#endif
      .def("__init__", [](phi::XPUEventHandle &self) {
#ifdef PADDLE_WITH_XPU
        new (&self) phi::XPUEventHandle();
#else
            PADDLE_THROW(common::errors::Unavailable(
                "Class XPUEvent can only be initialized on the XPU platform."));
#endif
      });
#ifdef PADDLE_WITH_XPU
  py::class_<phi::XPUCUDAStream>(m, "XPUCUDAStream", R"DOC(
      The handle of the XPU stream.

      Parameters:
          device(paddle.XPUPlace()|int|None, optional): The device which wanted to allocate the stream.
              If device is None or negative integer, device will be the current device.
              If device is positive integer, it must less than the device count. Default: None.
          priority(int|None, optional): The priority of stream. The priority can be 1(high) or 2(normal).
              If priority is None, the priority is 2(normal). Default: None.

      Examples:
          .. code-block:: python

              >>> # doctest: +REQUIRES(env:XPU)
              >>> import paddle
              >>> s1 = paddle.device.xpu.Stream(paddle.XPUPlace(0), 1)
              >>> s2 = paddle.device.xpu.Stream(0, 1)
              >>> s3 = paddle.device.xpu.Stream()

      )DOC")
      .def(
          "synchronize",
          [](phi::XPUCUDAStream &self) { self.Synchronize(); },
          R"DOC(
          Waits for stream tasks to complete.

          Examples:
              .. code-block:: python

                  >>> # doctest: +REQUIRES(env:GPU)
                  >>> import paddle
                  >>> s = paddle.device.cuda.Stream(paddle.CUDAPlace(0), 1)
                  >>> s.synchronize()

          )DOC")
      .def("__init__",
           [](phi::XPUCUDAStream &self, phi::XPUPlace *place, int priority) {
             if (priority != 1 && priority != 2) {
               PADDLE_THROW(common::errors::InvalidArgument(
                   "Priority should be 1(high) or 2(normal) "));
             }
             auto stream_flag =
                 phi::XPUCUDAStream::StreamFlag::kStreamNonBlocking;
             if (place == nullptr) {
               int curr_device_id = platform::GetXPUCurrentDeviceId();
               auto place_tmp = phi::XPUPlace(curr_device_id);
               new (&self)
                   phi::XPUCUDAStream(place_tmp, priority - 2, stream_flag);
             } else {
               new (&self)
                   phi::XPUCUDAStream(*place, priority - 2, stream_flag);
             }
           });
#endif
}
}  // namespace pybind
}  // namespace paddle
