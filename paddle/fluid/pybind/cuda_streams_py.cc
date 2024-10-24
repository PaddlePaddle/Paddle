// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/pybind/cuda_streams_py.h"

#include <string>
#include <vector>

#include "paddle/phi/api/profiler/event.h"
#include "paddle/phi/core/platform/device_event_base.h"

namespace py = pybind11;

namespace paddle {
namespace platform {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
phi::CUDAStream *get_current_stream(int device_id) {
  if (device_id == -1) {
    device_id = phi::backends::gpu::GetCurrentDeviceId();
  }
  auto *gpu_context = static_cast<const phi::GPUContext *>(
      DeviceContextPool::Instance().Get(GPUPlace(device_id)));
  return gpu_context->cuda_stream();
}

phi::CUDAStream *set_current_stream(phi::CUDAStream *stream) {
  auto *original_stream = get_current_stream(stream->place().GetDeviceId());
  auto *gpu_context = static_cast<phi::GPUContext *>(
      DeviceContextPool::Instance().Get(stream->place()));
  gpu_context->SetCUDAStream(stream, /*clear=*/false);
  return original_stream;
}
#endif
}  // namespace platform
namespace pybind {
void BindCudaStream(py::module *m_ptr) {
  auto &m = *m_ptr;

  // Bind Methods
  m.def(
      "_get_current_stream",
      [](int deviceId) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
        return platform::get_current_stream(deviceId);
#else
        PADDLE_THROW(
            common::errors::Unavailable("Paddle is not compiled with CUDA. "
                                        "Cannot visit device synchronize."));
#endif
      },
      py::return_value_policy::reference);

  m.def(
      "_set_current_stream",
      [](phi::CUDAStream *stream) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
        return platform::set_current_stream(stream);
#else
        PADDLE_THROW(
            common::errors::Unavailable("Paddle is not compiled with CUDA. "
                                        "Cannot visit device synchronize."));
#endif
      },
      py::return_value_policy::reference);

  m.def("_device_synchronize", [](int device_id) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    if (device_id == -1) {
      device_id = paddle::platform::GetCurrentDeviceId();
    }

    int curr_device_id = paddle::platform::GetCurrentDeviceId();
    paddle::platform::SetDeviceId(device_id);
#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_GPU_SUCCESS(hipDeviceSynchronize());
#else
    PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
#endif
    paddle::platform::SetDeviceId(curr_device_id);
#else
    PADDLE_THROW(common::errors::Unavailable(
        "Paddle is not compiled with CUDA. Cannot visit device synchronize."));
#endif
  });

  py::class_<phi::CUDAStream>(m, "CUDAStream", R"DOC(
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
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      .def(
          "wait_event",
          [](phi::CUDAStream &self, phi::CudaEvent &event) {
            self.WaitEvent(event.GetRawCudaEvent());
          },
          R"DOC(
          Makes all future work submitted to stream wait for all work captured in event.

          Parameters:
              event(CUDAEvent): The event to wait on.

          Examples:
              .. code-block:: python

                  >>> # doctest: +REQUIRES(env:GPU)
                  >>> import paddle
                  >>> s = paddle.device.cuda.Stream(paddle.CUDAPlace(0), 1)
                  >>> event = paddle.device.cuda.Event()
                  >>> s.wait_event(event)
          )DOC")
      .def(
          "wait_stream",
          [](phi::CUDAStream &self, phi::CUDAStream &stream) {
            phi::CudaEvent event;
            event.Record(stream.raw_stream());
            self.WaitEvent(event.GetRawCudaEvent());
          },
          R"DOC(
          Synchronizes with the given stream.

          Parameters:
              stream(CUDAStream): The stream to synchronize with.

          Examples:
              .. code-block:: python

                  >>> # doctest: +REQUIRES(env:GPU)
                  >>> import paddle
                  >>> s1 = paddle.device.cuda.Stream(paddle.CUDAPlace(0), 1)
                  >>> s2 = paddle.device.cuda.Stream(0, 1)
                  >>> s1.wait_stream(s2)

          )DOC")
      .def(
          "query",
          [](phi::CUDAStream &self) { return self.Query(); },
          R"DOC(
          Return the status whether if all operations in stream have completed.

          Returns: A boolean value.

          Examples:
              .. code-block:: python

                  >>> # doctest: +REQUIRES(env:GPU)
                  >>> import paddle
                  >>> s = paddle.device.cuda.Stream(paddle.CUDAPlace(0), 1)
                  >>> is_done = s.query()

          )DOC")
      .def(
          "synchronize",
          [](phi::CUDAStream &self) { self.Synchronize(); },
          R"DOC(
          Waits for stream tasks to complete.

          Examples:
              .. code-block:: python

                  >>> # doctest: +REQUIRES(env:GPU)
                  >>> import paddle
                  >>> s = paddle.device.cuda.Stream(paddle.CUDAPlace(0), 1)
                  >>> s.synchronize()

          )DOC")
      .def(
          "record_event",
          [](phi::CUDAStream &self, phi::CudaEvent *event) {
            if (event == nullptr) {
              event = new phi::CudaEvent();
            }
            event->Record(self.raw_stream());
            return event;
          },
          R"DOC(
          Record a CUDA event in the stream.

          Parameters:
              event(CUDAEvent, optional): The event to be record. If event is None, a new event is created.
                  Default: None.

          Returns:
              The record event.

          Examples:
              .. code-block:: python

                  >>> # doctest: +REQUIRES(env:GPU)
                  >>> import paddle
                  >>> s = paddle.device.cuda.Stream(paddle.CUDAPlace(0), 1)
                  >>> event = s.record_event()

          )DOC",
          py::arg("event") = nullptr)
      .def_property_readonly(
          "cuda_stream",
          [](phi::CUDAStream &self) {
            VLOG(10) << self.raw_stream();
            return reinterpret_cast<std::uintptr_t>(self.raw_stream());
          },
          R"DOC(
          return the raw cuda stream of type cudaStream_t as type int.

          Examples:
              .. code-block:: python

                  >>> # doctest: +REQUIRES(env:GPU)
                  >>> import paddle
                  >>> import ctypes
                  >>> cuda_stream = paddle.device.cuda.current_stream().cuda_stream
                  >>> print(cuda_stream)

                  >>> ptr = ctypes.c_void_p(cuda_stream)  # convert back to void*
                  >>> print(ptr)

          )DOC")
      .def_property_readonly(
          "place",
          [](phi::CUDAStream &self) { return phi::GPUPlace(self.place()); })
#endif
      .def(
          "__init__",
          [](phi::CUDAStream &self, phi::GPUPlace *place, int priority) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
            if (priority != 1 && priority != 2) {
              PADDLE_THROW(common::errors::InvalidArgument(
                  "Priority should be 1(high) or 2(normal) "));
            }

            auto stream_flag = phi::CUDAStream::StreamFlag::kStreamNonBlocking;
            if (place == nullptr) {
              int curr_device_id = platform::GetCurrentDeviceId();
              auto place_tmp = phi::GPUPlace(curr_device_id);
              new (&self) phi::CUDAStream(place_tmp, priority - 2, stream_flag);
            } else {
              // setting priority 1(high) and 2(normal) correspond to the actual
              // cuda stream priority -1 and 0.
              new (&self) phi::CUDAStream(*place, priority - 2, stream_flag);
            }
#else
            PADDLE_THROW(common::errors::Unavailable(
        "Class CUDAStream can only be initialized on the GPU platform."));
#endif
          },
          py::arg("device") = nullptr,
          py::arg("priority") = 2)
      .def(
          "__init__",
          [](phi::CUDAStream &self, int device, int priority) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
            if (priority != 1 && priority != 2) {
              PADDLE_THROW(common::errors::InvalidArgument(
                  "Priority should be 1(high) or 2(normal) "));
            }

            int device_count = platform::GetGPUDeviceCount();
            if (device < 0) {
              device = platform::GetCurrentDeviceId();
            }
            if (device >= device_count) {
              PADDLE_THROW(common::errors::InvalidArgument(
                  "The device id  must be inside [0, %d), but input device=%d.",
                  device_count,
                  device));
            }

            auto stream_flag = phi::CUDAStream::StreamFlag::kStreamNonBlocking;
            // setting priority 1(high) and 2(normal) correspond to the actual
            // cuda stream priority -1 and 0.
            new (&self) phi::CUDAStream(
                phi::GPUPlace(device), priority - 2, stream_flag);
#else
            PADDLE_THROW(common::errors::Unavailable(
        "Class CUDAStream can only be initialized on the GPU platform."));
#endif
          },
          py::arg("device") = -1,
          py::arg("priority") = 2)
      .def("__init__", [](phi::CUDAStream &self) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
        int device_id = platform::GetCurrentDeviceId();
        auto stream_flag = phi::CUDAStream::StreamFlag::kStreamNonBlocking;
        new (&self) phi::CUDAStream(
            phi::GPUPlace(device_id), /*priority=*/0, stream_flag);
#else
            PADDLE_THROW(common::errors::Unavailable(
        "Class CUDAStream can only be initialized on the GPU platform."));
#endif
      });

  py::class_<phi::CudaEvent>(m, "CUDAEvent", R"DOC(
      The handle of the CUDA event.

      Parameters:
          enable_timing(bool, optional): Whether the event will measure time. Default: False.
          blocking(bool, optional): Whether the wait() func will be blocking. Default: False;
          interprocess(bool, optional): Whether the event can be shared between processes. Default: False.

      Examples:
          .. code-block:: python

              >>> # doctest: +REQUIRES(env:GPU)
              >>> import paddle
              >>> event = paddle.device.cuda.Event()

      )DOC")
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      .def(
          "record",
          [](phi::CudaEvent &self, phi::CUDAStream *stream) {
            if (stream == nullptr) {
              stream = paddle::platform::get_current_stream(-1);
            }
            self.Record(stream->raw_stream());
          },
          R"DOC(
          Records the event in the given stream.

          Parameters:
              stream(CUDAStream, optional): The handle of CUDA stream. If None, the stream is the current stream. Default: None.

          Examples:
              .. code-block:: python

                  >>> # doctest: +REQUIRES(env:GPU)
                  >>> import paddle
                  >>> paddle.device.set_device('gpu')
                  >>> event = paddle.device.cuda.Event()
                  >>> event.record()

          )DOC",
          py::arg("stream") = nullptr)
      .def(
          "query",
          [](phi::CudaEvent &self) { return self.Query(); },
          R"DOC(
          Queries the event's status.

          Returns: A boolean which indicates all work currently captured by the event has been completed.

          Examples:
              .. code-block:: python

                  >>> # doctest: +REQUIRES(env:GPU)
                  >>> import paddle
                  >>> paddle.device.set_device('gpu')
                  >>> event = paddle.device.cuda.Event()
                  >>> is_done = event.query()

          )DOC")
      .def(
          "elapsed_time",
          [](phi::CudaEvent &self, phi::CudaEvent &end_event) {
            return self.ElapsedTime(&end_event);
          },
          R"DOC(
          Returns the time elapsed in milliseconds after the event was
          recorded and before the end_event was recorded.

          Returns: A int which indicates the elapsed time.

          Examples:
              .. code-block:: python

                  >>> # doctest: +REQUIRES(env:GPU)
                  >>> import paddle

                  >>> paddle.set_device('gpu')
                  >>> e1 = paddle.device.Event(enable_timing=True)
                  >>> e1.record()

                  >>> e2 = paddle.device.Event(enable_timing=True)
                  >>> e2.record()
                  >>> e1.elapsed_time(e2)

          )DOC")
      .def(
          "synchronize",
          [](phi::CudaEvent &self) { self.Synchronize(); },
          R"DOC(
            Waits for an event to complete.

            Examples:
                .. code-block:: python

                    >>> # doctest: +REQUIRES(env:GPU)
                    >>> import paddle
                    >>> paddle.device.set_device('gpu')
                    >>> event = paddle.device.cuda.Event()
                    >>> event.synchronize()

          )DOC")
#endif
      .def(
          "__init__",
          [](phi::CudaEvent &self,
             bool enable_timing,
             bool blocking,
             bool interprocess) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
            unsigned int flags = platform::GenerateDeviceEventFlag(
                enable_timing, blocking, interprocess);
            new (&self) phi::CudaEvent(flags);
#else
            PADDLE_THROW(common::errors::Unavailable(
                "Class CUDAEvent can only be initialized on the GPU "
                "platform."));

#endif
          },
          py::arg("enable_timing") = false,
          py::arg("blocking") = false,
          py::arg("interprocess") = false);
}

}  // namespace pybind
}  // namespace paddle
