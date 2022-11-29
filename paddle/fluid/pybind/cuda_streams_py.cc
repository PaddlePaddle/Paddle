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

#include "paddle/fluid/platform/device_event_base.h"
#include "paddle/fluid/platform/event.h"

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
            platform::errors::Unavailable("Paddle is not compiled with CUDA. "
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
            platform::errors::Unavailable("Paddle is not compiled with CUDA. "
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
    PADDLE_THROW(platform::errors::Unavailable(
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

            # required: gpu
            import paddle
            s1 = paddle.device.cuda.Stream(paddle.CUDAPlace(0), 1)
            s2 = paddle.device.cuda.Stream(0, 1)
            s3 = paddle.device.cuda.Stream()

  )DOC")
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      .def(
          "wait_event",
          [](phi::CUDAStream &self, paddle::platform::CudaEvent &event) {
            self.WaitEvent(event.GetRawCudaEvent());
          },
          R"DOC(
      Makes all future work submitted to stream wait for all work captured in event.

      Parameters:
        event(CUDAEvent): The event to wait on.

      Examples:
        .. code-block:: python

          # required: gpu
          import paddle
          s = paddle.device.cuda.Stream(paddle.CUDAPlace(0), 1)
          event = paddle.device.cuda.Event()
          s.wait_event(event)

           )DOC")
      .def(
          "wait_stream",
          [](phi::CUDAStream &self, phi::CUDAStream &stream) {
            paddle::platform::CudaEvent event;
            event.Record(stream.raw_stream());
            self.WaitEvent(event.GetRawCudaEvent());
          },
          R"DOC(
      Synchronizes with the given stream.

      Parameters:
        stream(CUDAStream): The stream to synchronize with.

      Examples:
        .. code-block:: python

            # required: gpu
            import paddle
            s1 = paddle.device.cuda.Stream(paddle.CUDAPlace(0), 1)
            s2 = paddle.device.cuda.Stream(0, 1)
            s1.wait_stream(s2)

           )DOC")
      .def(
          "query",
          [](phi::CUDAStream &self) { return self.Query(); },
          R"DOC(
      Return the status whether if all operations in stream have completed.

      Returns: A boolean value.

      Examples:
        .. code-block:: python

            # required: gpu
            import paddle
            s = paddle.device.cuda.Stream(paddle.CUDAPlace(0), 1)
            is_done = s.query()

           )DOC")
      .def(
          "synchronize",
          [](phi::CUDAStream &self) { self.Synchronize(); },
          R"DOC(
      Waits for stream tasks to complete.

      Examples:
        .. code-block:: python

            # required: gpu
            import paddle
            s = paddle.device.cuda.Stream(paddle.CUDAPlace(0), 1)
            s.synchronize()

           )DOC")
      .def(
          "record_event",
          [](phi::CUDAStream &self, paddle::platform::CudaEvent *event) {
            if (event == nullptr) {
              event = new paddle::platform::CudaEvent();
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
          The recored event.

      Examples:
        .. code-block:: python

            # required: gpu
            import paddle
            s = paddle.device.cuda.Stream(paddle.CUDAPlace(0), 1)
            event = s.record_event()

           )DOC",
          py::arg("event") = nullptr)
      .def_property_readonly(
          "cuda_stream",
          [](phi::CUDAStream &self) {
            VLOG(10) << self.raw_stream();
            return reinterpret_cast<std::uintptr_t>(self.raw_stream());
          },
          R"DOC(
      retrun the raw cuda stream of type cudaStream_t as type int.

      Examples:
        .. code-block:: python

            # required: gpu
            import paddle
            import ctypes
            cuda_stream = paddle.device.cuda.current_stream().cuda_stream
            print(cuda_stream)

            ptr = ctypes.c_void_p(cuda_stream)  # convert back to void*
            print(ptr)

           )DOC")
#endif
      .def(
          "__init__",
          [](phi::CUDAStream &self, platform::CUDAPlace *place, int priority) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
            if (priority != 1 && priority != 2) {
              PADDLE_THROW(platform::errors::InvalidArgument(
                  "Priority should be 1(high) or 2(normal) "));
            }
            auto prio = phi::CUDAStream::Priority(priority);
            auto stream_flag = phi::CUDAStream::StreamFlag::kStreamNonBlocking;

            if (place == nullptr) {
              int curr_device_id = platform::GetCurrentDeviceId();
              auto place_tmp = platform::CUDAPlace(curr_device_id);
              place = &place_tmp;
            }

            new (&self) phi::CUDAStream(*place, prio, stream_flag);
#else
            PADDLE_THROW(platform::errors::Unavailable(
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
              PADDLE_THROW(platform::errors::InvalidArgument(
                  "Priority should be 1(high) or 2(normal) "));
            }
            auto prio = phi::CUDAStream::Priority(priority);
            auto stream_flag = phi::CUDAStream::StreamFlag::kStreamNonBlocking;

            int device_count = platform::GetGPUDeviceCount();
            if (device < 0) {
              device = platform::GetCurrentDeviceId();
            }
            if (device >= device_count) {
              PADDLE_THROW(platform::errors::InvalidArgument(
                  "The device id  must be inside [0, %d), but input device=%d.",
                  device_count,
                  device));
            }

            new (&self)
                phi::CUDAStream(platform::CUDAPlace(device), prio, stream_flag);
#else
            PADDLE_THROW(platform::errors::Unavailable(
        "Class CUDAStream can only be initialized on the GPU platform."));
#endif
          },
          py::arg("device") = -1,
          py::arg("priority") = 2)
      .def("__init__", [](phi::CUDAStream &self) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
        auto prio = phi::CUDAStream::Priority::kNormal;
        auto stream_flag = phi::CUDAStream::StreamFlag::kStreamNonBlocking;

        int device_id = platform::GetCurrentDeviceId();

        new (&self)
            phi::CUDAStream(platform::CUDAPlace(device_id), prio, stream_flag);
#else
            PADDLE_THROW(platform::errors::Unavailable(
        "Class CUDAStream can only be initialized on the GPU platform."));
#endif
      });

  py::class_<paddle::platform::CudaEvent>(m, "CUDAEvent", R"DOC(
      The handle of the CUDA event.

      Parameters:
        enable_timing(bool, optional): Whether the event will measure time. Default: False.
        blocking(bool, optional): Whether the wait() func will be blocking. Default: False;
        interprocess(bool, optional): Whether the event can be shared between processes. Default: False.

      Examples:
        .. code-block:: python

            # required: gpu
            import paddle
            event = paddle.device.cuda.Event()

  )DOC")
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      .def(
          "record",
          [](paddle::platform::CudaEvent &self, phi::CUDAStream *stream) {
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

              # required: gpu
              import paddle
              event = paddle.device.cuda.Event()
              event.record()

        )DOC",
          py::arg("stream") = nullptr)
      .def(
          "query",
          [](paddle::platform::CudaEvent &self) { return self.Query(); },
          R"DOC(
          Queries the event's status.

          Returns: A boolean which indicates all work currently captured by the event has been completed.

          Examples:
            .. code-block:: python

                # required: gpu
                import paddle
                event = paddle.device.cuda.Event()
                is_done = event.query()

           )DOC")
      .def(
          "synchronize",
          [](paddle::platform::CudaEvent &self) { self.Synchronize(); },
          R"DOC(
            Waits for an event to complete.

            Examples:
              .. code-block:: python

                # required: gpu
                import paddle
                event = paddle.device.cuda.Event()
                event.synchronize()

           )DOC")
#endif
      .def(
          "__init__",
          [](paddle::platform::CudaEvent &self,
             bool enable_timing,
             bool blocking,
             bool interprocess) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
            unsigned int flags = platform::GenerateDeviceEventFlag(
                enable_timing, blocking, interprocess);
            new (&self) paddle::platform::CudaEvent(flags);
#else
            PADDLE_THROW(platform::errors::Unavailable(
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
