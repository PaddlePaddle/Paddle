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

#include "paddle/phi/backends/device_manager.h"
#include "paddle/phi/backends/event.h"
#include "paddle/phi/backends/stream.h"
#include "paddle/phi/core/platform/device_context.h"

namespace py = pybind11;

namespace paddle::pybind {
void BindCustomDevicePy(py::module *m_ptr) {
  auto &m = *m_ptr;
  // Bind Methods
  m.def("_get_device_min_chunk_size", [](const std::string &device_type) {
    auto place = phi::CustomPlace(device_type);
    return phi::DeviceManager::GetMinChunkSize(place);
  });
  m.def(
      "_get_device_total_memory",
      [](const std::string &device_type, int device_id) {
        auto place = phi::CustomPlace(
            device_type,
            device_id == -1 ? phi::DeviceManager::GetDevice(device_type)
                            : device_id);
        size_t total = 0, free = 0;
        phi::DeviceManager::MemoryStats(place, &total, &free);
        return total;
      },
      py::arg("device_type"),
      py::arg("device_id") = -1);
  m.def(
      "_get_current_custom_device_stream",
      [](const std::string &device_type, int device_id) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
        auto place = phi::CustomPlace(
            device_type,
            device_id == -1 ? phi::DeviceManager::GetDevice(device_type)
                            : device_id);

        return static_cast<const phi::CustomContext *>(
                   phi::DeviceContextPool::Instance().Get(place))
            ->GetStream();
#else
        PADDLE_THROW(common::errors::Unavailable(
            "Paddle is not compiled with CustomDevice. "
            "Cannot visit _get_current_custom_device_stream."));
#endif
      },
      py::return_value_policy::reference,
      py::arg("device_type"),
      py::arg("device_id") = -1);
  m.def(
      "_set_current_custom_device_stream",
      [](const std::string &device_type,
         int device_id,
         std::shared_ptr<phi::stream::Stream> stream) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
        auto place = phi::CustomPlace(
            device_type,
            device_id == -1 ? phi::DeviceManager::GetDevice(device_type)
                            : device_id);
        static_cast<phi::CustomContext *>(
            phi::DeviceContextPool::Instance().Get(place))
            ->SetStream(stream);
        return stream;
#else
        PADDLE_THROW(common::errors::Unavailable(
            "Paddle is not compiled with CustomDevice. "
            "Cannot visit _set_current_custom_device_stream."));
#endif
      },
      py::arg("device_type"),
      py::arg("device_id") = -1,
      py::arg("stream") = nullptr);
  m.def("_synchronize_custom_device",
        [](const std::string &device_type, int device_id) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
          auto place = phi::CustomPlace(
              device_type,
              device_id == -1 ? phi::DeviceManager::GetDevice(device_type)
                              : device_id);
          phi::DeviceManager::SynchronizeDevice(place);
#else
        PADDLE_THROW(common::errors::Unavailable(
            "Paddle is not compiled with CustomDevice. "
            "Cannot visit _synchronize_custom_device."));
#endif
        });

  py::class_<phi::stream::Stream, std::shared_ptr<phi::stream::Stream>>(
      m, "CustomDeviceStream", R"DOC(
      The handle of the custom device stream.

      Parameters:
          device(paddle.CustomPlace()|str): The device which wanted to allocate the stream.
          device_id(int, optional): The id of the device which wanted to allocate the stream.
              If device is None or negative integer, device will be the current device.
              If device is positive integer, it must less than the device count. Default: None.
          priority(int|None, optional): The priority of stream. The priority can be 1(high) or 2(normal).
              If priority is None, the priority is 2(normal). Default: None.
          blocking(int|None, optional): Whether the stream is executed synchronously. Default: False.

      Examples:
          .. code-block:: python

              >>> # doctest: +REQUIRES(env:CUSTOM_DEVICE)
              >>> import paddle
              >>> s3 = paddle.device.custom.Stream('custom_cpu')
              >>> s2 = paddle.device.custom.Stream('custom_cpu', 0)
              >>> s1 = paddle.device.custom.Stream(paddle.CustomPlace('custom_cpu'))
              >>> s1 = paddle.device.custom.Stream(paddle.CustomPlace('custom_cpu'), 1)
              >>> s1 = paddle.device.custom.Stream(paddle.CustomPlace('custom_cpu'), 1, True)

      )DOC")
      .def(
          "__init__",
          [](phi::stream::Stream &self,
             const phi::CustomPlace &place,
             int priority,
             bool blocking) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
            new (&self) phi::stream::Stream();
            self.Init(
                place,
                static_cast<phi::stream::Stream::Priority>(priority),
                static_cast<phi::stream::Stream::Flag>(
                    blocking ? phi::stream::Stream::Flag::kDefaultFlag
                             : phi::stream::Stream::Flag::kStreamNonBlocking));
#else
        PADDLE_THROW(common::errors::Unavailable(
            "Paddle is not compiled with CustomDevice. "
            "Cannot visit CustomDeviceStream."));
#endif
          },
          py::arg("device"),
          py::arg("priority") = 2,
          py::arg("blocking") = false)
      .def(
          "__init__",
          [](phi::stream::Stream &self,
             const std::string &device_type,
             int device_id,
             int priority,
             bool blocking) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
            new (&self) phi::stream::Stream();
            self.Init(
                phi::CustomPlace(
                    device_type,
                    device_id == -1 ? phi::DeviceManager::GetDevice(device_type)
                                    : device_id),
                static_cast<phi::stream::Stream::Priority>(priority),
                static_cast<phi::stream::Stream::Flag>(
                    blocking ? phi::stream::Stream::Flag::kDefaultFlag
                             : phi::stream::Stream::Flag::kStreamNonBlocking));
#else
        PADDLE_THROW(common::errors::Unavailable(
            "Paddle is not compiled with CustomDevice. "
            "Cannot visit CustomDeviceStream."));
#endif
          },
          py::arg("device"),
          py::arg("device_id") = -1,
          py::arg("priority") = 2,
          py::arg("blocking") = false)
      .def(
          "wait_event",
          [](const phi::stream::Stream &self, phi::event::Event *event) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
            self.WaitEvent(event);
#else
        PADDLE_THROW(common::errors::Unavailable(
            "Paddle is not compiled with CustomDevice. "
            "Cannot visit CustomDeviceStream."));
#endif
          },
          R"DOC(
          Makes all future work submitted to stream wait for all work captured in event.

          Parameters:
              event(CustomDeviceEvent): The event to wait on.

          Examples:
              .. code-block:: python

                  >>> # doctest: +REQUIRES(env:CUSTOM_DEVICE)
                  >>> import paddle
                  >>> place = paddle.CustomPlace('custom_cpu', 0)
                  >>> s = paddle.device.custom.Stream(place)
                  >>> event = paddle.device.custom.Event(place)
                  >>> s.wait_event(event)

          )DOC")
      .def(
          "wait_stream",
          [](const phi::stream::Stream &self, phi::stream::Stream *other) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
            phi::event::Event event;
            event.Init(self.GetPlace());
            event.Record(other);
            self.WaitEvent(&event);
#else
        PADDLE_THROW(common::errors::Unavailable(
            "Paddle is not compiled with CustomDevice. "
            "Cannot visit CustomDeviceStream."));
#endif
          },
          R"DOC(
          Synchronizes with the given stream.

          Parameters:
              stream(CUDAStream): The stream to synchronize with.

          Examples:
              .. code-block:: python

                  >>> # doctest: +REQUIRES(env:CUSTOM_DEVICE)
                  >>> import paddle
                  >>> place = paddle.CustomPlace('custom_cpu', 0)
                  >>> s1 = paddle.device.custom.Stream(place)
                  >>> s2 = paddle.device.custom.Stream(place)
                  >>> s1.wait_stream(s2)

          )DOC")
      .def(
          "query",
          [](const phi::stream::Stream &self) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
            return self.Query();
#else
        PADDLE_THROW(common::errors::Unavailable(
            "Paddle is not compiled with CustomDevice. "
            "Cannot visit CustomDeviceStream."));
#endif
          },
          R"DOC(
          Return the status whether if all operations in stream have completed.

          Returns:
              A boolean value.

          Examples:
              .. code-block:: python

                  >>> # doctest: +REQUIRES(env:CUSTOM_DEVICE)
                  >>> import paddle
                  >>> place = paddle.CustomPlace('custom_cpu', 0)
                  >>> s = paddle.device.custom.Stream(place)
                  >>> is_done = s.query()

          )DOC")
      .def(
          "synchronize",
          [](const phi::stream::Stream &self) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
            self.Synchronize();
#else
        PADDLE_THROW(common::errors::Unavailable(
            "Paddle is not compiled with CustomDevice. "
            "Cannot visit CustomDeviceStream."));
#endif
          },
          R"DOC(
          Waits for stream tasks to complete.

          Examples:
              .. code-block:: python

                  >>> # doctest: +REQUIRES(env:CUSTOM_DEVICE)
                  >>> import paddle
                  >>> place = paddle.CustomPlace('custom_cpu', 0)
                  >>> s = paddle.device.custom.Stream(place)
                  >>> s.synchronize()

          )DOC")
      .def(
          "record_event",
          [](const phi::stream::Stream &self, phi::event::Event *event) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
            if (event == nullptr) {
              event = new phi::event::Event;
              event->Init(self.GetPlace());
            }
            event->Record(&self);
            return event;
#else
        PADDLE_THROW(common::errors::Unavailable(
            "Paddle is not compiled with CustomDevice. "
            "Cannot visit CustomDeviceStream."));
#endif
          },
          R"DOC(
          Record an event in the stream.

          Parameters:
              event(CustomDeviceEvent, optional): The event to be record. If event is None, a new event is created.
                  Default: None.

          Returns:
              The record event.

          Examples:
              .. code-block:: python

                  >>> # doctest: +REQUIRES(env:CUSTOM_DEVICE)
                  >>> import paddle
                  >>> place = paddle.CustomPlace('custom_cpu', 0)
                  >>> s = paddle.device.custom.Stream(place)
                  >>> event = s.record_event()

          )DOC",
          py::arg("event") = nullptr)
      .def_property_readonly(
          "raw_stream",
          [](const phi::stream::Stream &self) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
            VLOG(10) << self.raw_stream();
            return reinterpret_cast<std::uintptr_t>(self.raw_stream());
#else
        PADDLE_THROW(common::errors::Unavailable(
            "Paddle is not compiled with CustomDevice. "
            "Cannot visit CustomDeviceStream."));
#endif
          },
          R"DOC(
          return the raw stream of type CustomDeviceStream as type int.

          Examples:
            .. code-block:: python

                >>> # doctest: +REQUIRES(env:CUSTOM_DEVICE)
                >>> import paddle
                >>> import ctypes
                >>> stream  = paddle.device.custom.current_stream().raw_stream
                >>> print(stream)

                >>> ptr = ctypes.c_void_p(stream)  # convert back to void*
                >>> print(ptr)

          )DOC")
      .def_property_readonly("place", [](const phi::stream::Stream &self) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
        return reinterpret_cast<const phi::CustomPlace &>(self.GetPlace());
#else
        PADDLE_THROW(common::errors::Unavailable(
            "Paddle is not compiled with CustomDevice. "
            "Cannot visit CustomDeviceStream."));
#endif
      });

  py::class_<phi::event::Event, std::shared_ptr<phi::event::Event>>(
      m, "CustomDeviceEvent", R"DOC(
      The handle of the custom device event.

      Parameters:
          device(paddle.CustomPlace()|str): The device which wanted to allocate the stream.
          device_id(int, optional): The id of the device which wanted to allocate the stream.
              If device is None or negative integer, device will be the current device.
              If device is positive integer, it must less than the device count. Default: None.
          enable_timing(bool, optional): Whether the event will measure time. Default: False.
          blocking(bool, optional): Whether the wait() func will be blocking. Default: False.
          interprocess(bool, optional): Whether the event can be shared between processes. Default: False.

      Examples:
          .. code-block:: python

              >>> # doctest: +REQUIRES(env:CUSTOM_DEVICE)
              >>> import paddle
              >>> place = paddle.CustomPlace('custom_cpu', 0)
              >>> event = paddle.device.custom.Event(place)

      )DOC")
      .def(
          "__init__",
          [](phi::event::Event &self,
             const phi::CustomPlace &place,
             bool enable_timing,
             bool blocking,
             bool interprocess) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
            auto flag = static_cast<phi::event::Event::Flag>(
                static_cast<uint32_t>(
                    enable_timing ? 0
                                  : phi::event::Event::Flag::DisableTiming) |
                static_cast<uint32_t>(
                    !blocking ? 0 : phi::event::Event::Flag::BlockingSync) |
                static_cast<uint32_t>(
                    !interprocess ? 0 : phi::event::Event::Flag::Interprocess)

            );
            new (&self) phi::event::Event();
            self.Init(place, flag);
#else
        PADDLE_THROW(common::errors::Unavailable(
            "Paddle is not compiled with CustomDevice. "
            "Cannot visit CustomDeviceEvent."));
#endif
          },
          py::arg("device"),
          py::arg("enable_timing") = false,
          py::arg("blocking") = false,
          py::arg("interprocess") = false)
      .def(
          "__init__",
          [](phi::event::Event &self,
             const std::string &device_type,
             int device_id,
             bool enable_timing,
             bool blocking,
             bool interprocess) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
            auto flag = static_cast<phi::event::Event::Flag>(
                static_cast<uint32_t>(
                    enable_timing ? 0
                                  : phi::event::Event::Flag::DisableTiming) |
                static_cast<uint32_t>(
                    !blocking ? 0 : phi::event::Event::Flag::BlockingSync) |
                static_cast<uint32_t>(
                    !interprocess ? 0 : phi::event::Event::Flag::Interprocess)

            );
            new (&self) phi::event::Event();
            self.Init(
                phi::CustomPlace(
                    device_type,
                    device_id == -1 ? phi::DeviceManager::GetDevice(device_type)
                                    : device_id),
                flag);
#else
        PADDLE_THROW(common::errors::Unavailable(
            "Paddle is not compiled with CustomDevice. "
            "Cannot visit CustomDeviceEvent."));
#endif
          },
          py::arg("device"),
          py::arg("device_id") = -1,
          py::arg("enable_timing") = false,
          py::arg("blocking") = false,
          py::arg("interprocess") = false)
      .def(
          "record",
          [](phi::event::Event &self, phi::stream::Stream *stream) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
            if (stream == nullptr) {
              stream =
                  static_cast<const phi::CustomContext *>(
                      phi::DeviceContextPool::Instance().Get(self.GetPlace()))
                      ->GetStream()
                      .get();
            }
            self.Record(stream);
#else
        PADDLE_THROW(common::errors::Unavailable(
            "Paddle is not compiled with CustomDevice. "
            "Cannot visit CustomDeviceEvent."));
#endif
          },
          R"DOC(
          Records the event in the given stream.

          Parameters:
              stream(CustomDeviceStream, optional): The handle of custom device stream. If None, the stream is the current stream. Default: None.

          Examples:
              .. code-block:: python

                  >>> # doctest: +REQUIRES(env:CUSTOM_DEVICE)
                  >>> import paddle
                  >>> place = paddle.CustomPlace('custom_cpu', 0)
                  >>> event = paddle.device.custom.Event(place)
                  >>> event.record()

          )DOC")
      .def(
          "query",
          [](const phi::event::Event &self) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
            return self.Query();
#else
        PADDLE_THROW(common::errors::Unavailable(
            "Paddle is not compiled with CustomDevice. "
            "Cannot visit CustomDeviceEvent."));
#endif
          },
          R"DOC(
          Queries the event's status.

          Returns:
              A boolean which indicates all work currently captured by the event has been completed.

          Examples:
              .. code-block:: python

                  >>> # doctest: +REQUIRES(env:CUSTOM_DEVICE)
                  >>> import paddle
                  >>> place = paddle.CustomPlace('custom_cpu', 0)
                  >>> event = paddle.device.cuda.Event(place)
                  >>> is_done = event.query()

          )DOC")
      .def(
          "synchronize",
          [](const phi::event::Event &self) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
            self.Synchronize();
#else
        PADDLE_THROW(common::errors::Unavailable(
            "Paddle is not compiled with CustomDevice. "
            "Cannot visit CustomDeviceEvent."));
#endif
          },
          R"DOC(
            Waits for an event to complete.

            Examples:
                .. code-block:: python

                    >>> # doctest: +REQUIRES(env:CUSTOM_DEVICE)
                    >>> import paddle
                    >>> place = paddle.CustomPlace('custom_cpu', 0)
                    >>> event = paddle.device.custom.Event(place)
                    >>> event.synchronize()

          )DOC")
      .def_property_readonly(
          "raw_event",
          [](const phi::event::Event &self) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
            VLOG(10) << self.raw_event();
            return reinterpret_cast<std::uintptr_t>(self.raw_event());
#else
        PADDLE_THROW(common::errors::Unavailable(
            "Paddle is not compiled with CustomDevice. "
            "Cannot visit CustomDeviceEvent."));
#endif
          },
          R"DOC(
          return the raw event of type CustomDeviceEvent as type int.

          Examples:
              .. code-block:: python

                  >>> # doctest: +REQUIRES(env:CUSTOM_DEVICE)
                  >>> import paddle
                  >>> import ctypes
                  >>> place = paddle.CustomPlace('custom_cpu', 0)
                  >>> event = paddle.device.custom.Event(place)
                  >>> raw_event = event.raw_event
                  >>> print(raw_event)

                  >>> ptr = ctypes.c_void_p(raw_event)  # convert back to void*
                  >>> print(ptr)

          )DOC")
      .def_property_readonly("place", [](const phi::event::Event &self) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
        return reinterpret_cast<const phi::CustomPlace &>(self.GetPlace());
#else
        PADDLE_THROW(common::errors::Unavailable(
            "Paddle is not compiled with CustomDevice. "
            "Cannot visit CustomDeviceEvent."));
#endif
      });
}
}  // namespace paddle::pybind
