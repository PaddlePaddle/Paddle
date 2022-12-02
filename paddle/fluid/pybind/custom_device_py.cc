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

#include "paddle/fluid/pybind/custom_device_py.h"

#include <string>
#include <vector>

#include "paddle/fluid/platform/device_context.h"
#include "paddle/phi/backends/device_manager.h"
#include "paddle/phi/backends/event.h"
#include "paddle/phi/backends/stream.h"

namespace py = pybind11;

namespace paddle {
namespace pybind {
void BindCustomDevicePy(py::module *m_ptr) {
  auto &m = *m_ptr;
  // Bind Methods
  m.def("_custom_device_count", [](const std::string &device_type) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
    return phi::DeviceManager::GetDeviceCount(device_type);
#else
        PADDLE_THROW(platform::errors::Unavailable(
            "Paddle is not compiled with CustomDevice. "
            "Cannot visit _custom_device_count."));
#endif
  });
  m.def("_get_current_custom_device", [](const std::string &device_type) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
    return phi::DeviceManager::GetDevice(device_type);
#else
        PADDLE_THROW(platform::errors::Unavailable(
            "Paddle is not compiled with CustomDevice. "
            "Cannot visit _get_current_custom_device."));
#endif
  });
  m.def("_set_current_custom_device", [](const phi::CustomPlace &place) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
    phi::DeviceManager::SetDevice(place);
#else
        PADDLE_THROW(platform::errors::Unavailable(
            "Paddle is not compiled with CustomDevice. "
            "Cannot visit _set_current_custom_device."));
#endif
  });
  m.def("_synchronize_custom_device", [](const phi::CustomPlace &place) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
    phi::DeviceManager::SynchronizeDevice(place);
#else
        PADDLE_THROW(platform::errors::Unavailable(
            "Paddle is not compiled with CustomDevice. "
            "Cannot visit _synchronize_custom_device."));
#endif
  });
  m.def(
      "_get_current_custom_device_stream",
      [](const phi::CustomPlace &place) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
        return static_cast<const phi::CustomContext *>(
                   paddle::platform::DeviceContextPool::Instance().Get(place))
            ->GetStream();
#else
        PADDLE_THROW(platform::errors::Unavailable(
            "Paddle is not compiled with CustomDevice. "
            "Cannot visit _get_current_custom_device_stream."));
#endif
      },
      py::return_value_policy::reference);
  m.def("_set_current_custom_device_stream",
        [](const phi::CustomPlace &place,
           std::shared_ptr<phi::stream::Stream> stream) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
          static_cast<phi::CustomContext *>(
              paddle::platform::DeviceContextPool::Instance().Get(place))
              ->SetStream(stream);
#else
        PADDLE_THROW(platform::errors::Unavailable(
            "Paddle is not compiled with CustomDevice. "
            "Cannot visit _set_current_custom_device_stream."));
#endif
        });

  py::class_<phi::stream::Stream>(m, "CustomDeviceStream", R"DOC(
      The handle of the custom device stream.

      Parameters:
        device(paddle.CUDAPlace()|int|None, optional): The device which wanted to allocate the stream.
        If device is None or negative integer, device will be the current device.
        If device is positive integer, it must less than the device count. Default: None.

        priority(int|None, optional): The priority of stream. The priority can be 1(high) or 2(normal).
        If priority is None, the priority is 2(normal). Default: None.

      Examples:
        .. code-block:: python

            # required: custom_device
            import paddle
            s1 = paddle.device.custom.Stream(paddle.CUDAPlace(0), 1)
            s2 = paddle.device.custom.Stream(0, 1)
            s3 = paddle.device.custom.Stream()

  )DOC")
      .def("get_place",
           [](const phi::stream::Stream &self) { return self.GetPlace(); })
      .def(
          "wait_event",
          [](const phi::stream::Stream &self, phi::event::Event *event) {
            self.WaitEvent(event);
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
          [](const phi::stream::Stream &self, phi::stream::Stream *other) {
            phi::event::Event event;
            event.Init(self.GetPlace());
            event.Record(other);
            self.WaitEvent(&event);
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
          [](const phi::stream::Stream &self) { return self.Query(); },
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
          [](const phi::stream::Stream &self) { self.Synchronize(); },
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
          [](const phi::stream::Stream &self, phi::event::Event *event) {
            if (event == nullptr) {
              event = new phi::event::Event;
              event->Init(self.GetPlace());
            }
            event->Record(&self);
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
          "raw_stream",
          [](const phi::stream::Stream &self) {
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

           )DOC");

  py::class_<phi::event::Event>(m, "CustomDeviceEvent", R"DOC(
      The handle of the custom device event.

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
      .def("get_place",
           [](const phi::event::Event &self) { return self.GetPlace(); })
      .def(
          "record",
          [](phi::event::Event &self, phi::stream::Stream *stream) {
            self.Record(stream);
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

        )DOC")
      .def(
          "query",
          [](const phi::event::Event &self) { return self.Query(); },
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
          [](const phi::event::Event &self) { self.Synchronize(); },
          R"DOC(
            Waits for an event to complete.

            Examples:
              .. code-block:: python

                # required: gpu
                import paddle
                event = paddle.device.cuda.Event()
                event.synchronize()

           )DOC")
      .def_property_readonly(
          "raw_event",
          [](const phi::event::Event &self) {
            VLOG(10) << self.raw_event();
            return reinterpret_cast<std::uintptr_t>(self.raw_event());
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

           )DOC");
}
}  // namespace pybind
}  // namespace paddle
