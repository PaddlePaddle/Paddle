/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
// disable numpy compile error
#include <Python.h>

#include <set>
#include <string>
#include <vector>

#pragma GCC diagnostic ignored "-Wattributes"
#include "paddle/fluid/eager/accumulation/accumulation_node.h"
#include "paddle/fluid/eager/api/all.h"
#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/utils.h"
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/pybind/eager.h"
#include "paddle/fluid/pybind/eager_utils.h"
#include "paddle/fluid/pybind/exception.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "pybind11/detail/internals.h"
#include "pybind11/pytypes.h"
#pragma GCC diagnostic ignored "-Wwrite-strings"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/phi/backends/stream.h"
#include "paddle/phi/core/cuda_stream.h"

namespace paddle {
namespace pybind {

namespace py = ::pybind11;

PyTypeObject* p_streambase_type;
PyTypeObject* p_eventbase_type;
extern PyTypeObject* p_tensor_type;

PyObject* EventBaseNew(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
  EAGER_TRY
  bool enable_timing = false;
  bool blocking = false;
  bool interprocess = false;

  static char* kwlist[] = {
      "enable_timing", "blocking", "interprocess", nullptr};
  if (!PyArg_ParseTupleAndKeywords(args,
                                   kwargs,
                                   "|bbb",
                                   kwlist,
                                   &enable_timing,
                                   &blocking,
                                   &interprocess)) {
    return nullptr;
  }

  PyObject* obj = type->tp_alloc(type, 0);
  if (obj) {
    auto v = reinterpret_cast<EventBaseObject*>(obj);
    auto place = egr::Controller::Instance().GetExpectedPlace();
    v->event = new paddle::platform::DeviceEvent(
        place,
        platform::GenerateDeviceEventFlag(
            enable_timing, blocking, interprocess));
  }
  return obj;
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static void EventBaseDealloc(EventBaseObject* self) {
  Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

static PyObject* eventbase_method_record(EventBaseObject* self,
                                         StreamBaseObject* stream) {
  EAGER_TRY
  self->event->Record(stream->stream);
  RETURN_PY_NONE
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* eventbase_method_wait(EventBaseObject* self,
                                       StreamBaseObject* stream) {
  EAGER_TRY {
    eager_gil_scoped_release guard;
    self->event->Wait(
        paddle::platform::Place2DeviceType(stream->stream->GetPlace()),
        stream->stream);
  }
  RETURN_PY_NONE
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* eventbase_method_query(EventBaseObject* self) {
  EAGER_TRY
  return ToPyObject(self->event->Query());
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* eventbase_method_elapsed_time(EventBaseObject* self,
                                               EventBaseObject* other) {
  EAGER_TRY
  // TODO(wanghuancoder) elapsed_time
  // return ToPyObject(self->event->elapsed_time(other->event));//double
  return ToPyObject(0);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* eventbase_method_synchronize(EventBaseObject* self) {
  EAGER_TRY {
    eager_gil_scoped_release guard;
    self->event->Finish();
  }
  RETURN_PY_NONE
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* eventbaseobject_get_event(EventBaseObject* self, PyObject*) {
  EAGER_TRY
  return ToPyObject(self->event->GetEvent().get());
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* eventbaseobject_get_device(EventBaseObject* self, PyObject*) {
  EAGER_TRY
  return ToPyObject(self->event->place());
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyMethodDef eventbase_methods[] = {
    {"record",
     (PyCFunction)(void (*)(void))eventbase_method_record,
     METH_O,
     NULL},
    {"wait", (PyCFunction)(void (*)(void))eventbase_method_wait, METH_O, NULL},
    {"query",
     (PyCFunction)(void (*)(void))eventbase_method_query,
     METH_NOARGS,
     NULL},
    {"elapsed_time",
     (PyCFunction)(void (*)(void))eventbase_method_elapsed_time,
     METH_O,
     NULL},
    {"synchronize",
     (PyCFunction)(void (*)(void))eventbase_method_synchronize,
     METH_NOARGS,
     NULL},
    {NULL, NULL, 0, NULL}};

struct PyGetSetDef eventbase_properties[] {
  {"device", (getter)eventbaseobject_get_device, nullptr, nullptr, nullptr},
      {"event", (getter)eventbaseobject_get_event, nullptr, nullptr, nullptr}, {
    nullptr, nullptr, nullptr, nullptr, nullptr
  }
};

void BindEagerDeviceEvent(PyObject* module) {
  auto heap_type = reinterpret_cast<PyHeapTypeObject*>(
      PyType_Type.tp_alloc(&PyType_Type, 0));
  heap_type->ht_name = ToPyObject("EventBase");
  heap_type->ht_qualname = ToPyObject("EventBase");
  auto type = &heap_type->ht_type;
  type->tp_name = "EventBase";
  type->tp_basicsize = sizeof(EventBaseObject);
  type->tp_dealloc = (destructor)EventBaseDealloc;
  type->tp_methods = eventbase_methods;
  type->tp_getset = eventbase_properties;
  type->tp_new = (newfunc)EventBaseNew;
  Py_INCREF(&PyBaseObject_Type);
  type->tp_base = reinterpret_cast<PyTypeObject*>(&PyBaseObject_Type);
  type->tp_flags |= Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
#if PY_VERSION_HEX >= 0x03050000
  type->tp_as_async = &heap_type->as_async;
#endif
  p_eventbase_type = type;

  if (PyType_Ready(type) < 0) {
    PADDLE_THROW(platform::errors::Fatal(
        "Init Paddle error in BindEagerDeviceEvent(PyType_Ready)."));
    return;
  }

  Py_INCREF(type);
  if (PyModule_AddObject(
          module, "EventBase", reinterpret_cast<PyObject*>(type)) < 0) {
    Py_DECREF(type);
    Py_DECREF(module);
    PADDLE_THROW(platform::errors::Fatal(
        "Init Paddle error in BindEagerDeviceEvent(PyModule_AddObject)."));
    return;
  }
}

PyObject* StreamBaseNew(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
  EAGER_TRY
  auto p = CastPyArg2Place(PyTuple_GET_ITEM(args, 0), 0);

  PyObject* obj = type->tp_alloc(type, 0);
  if (obj) {
    auto v = reinterpret_cast<StreamBaseObject*>(obj);
    v->owned_stream_ = true;
    if (platform::is_cpu_place(p)) {
      PADDLE_THROW(platform::errors::Unimplemented(
          "CPUPlace is not supporte multi stream."));
    } else if (platform::is_gpu_place(p)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      v->stream =
          paddle::platform::CreateDeviceContext<phi::GPUContext>(p, true)
              .release();
      int priority = CastPyArg2AttrInt(PyTuple_GET_ITEM(args, 1), 1);
      auto priority_framework = phi::CUDAStream::Priority(2);
      if (priority == -1) {
        priority_framework = phi::CUDAStream::Priority(1);
      }
      auto* gpu_ctx = dynamic_cast<phi::GPUContext*>(v->stream);
      gpu_ctx->SetCUDAStream(new phi::CUDAStream(p, priority_framework), true);
#else
      PADDLE_THROW(
          platform::errors::Unimplemented("CUDAPlace is not supported. Please "
                                          "re-compile with WITH_GPU option."));
#endif
      //     } else if (platform::is_xpu_place(p)) {
      // #ifdef PADDLE_WITH_XPU
      //       v->stream =
      //       paddle::platform::CreateDeviceContext<XPUDeviceContext>(p,
      //       true).release();
      // #else
      //       PADDLE_THROW(
      //           platform::errors::Unimplemented("XPUPlace is not supported.
      //           Please "
      //                                           "re-compile with WITH_XPU
      //                                           option."));
      // #endif
      //     } else if (platform::is_mlu_place(p)) {
      // #ifdef PADDLE_WITH_MLU
      //       v->stream =
      //       paddle::platform::CreateDeviceContext<MLUDeviceContext>(p,
      //       true).release();
      // #else
      //       PADDLE_THROW(
      //           platform::errors::Unimplemented("MLUPlace is not supported.
      //           Please "
      //                                           "re-compile with WITH_MLU
      //                                           option."));
      // #endif
      //     } else if (platform::is_ipu_place(p)) {
      // #ifdef PADDLE_WITH_IPU
      //       v->stream =
      //       paddle::platform::CreateDeviceContext<IPUDeviceContext>(p,
      //       true).release();
      // #else
      //       PADDLE_THROW(
      //           platform::errors::Unimplemented("IPUPlace is not supported.
      //           Please "
      //                                           "re-compile with WITH_IPU
      //                                           option."));
      // #endif
      //     } else if (platform::is_npu_place(p)) {
      // #ifdef PADDLE_WITH_ASCEND_CL
      //       v->stream =
      //       paddle::platform::CreateDeviceContext<NPUDeviceContext>(p,
      //       true).release();
      // #else
      //       PADDLE_THROW(platform::errors::Unimplemented(
      //           "NPUPlace is not supported. Please "
      //           "re-compile with WITH_ASCEND_CL option."));
      // #endif
    } else if (platform::is_custom_place(p)) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
      v->stream = paddle::platform::CreateDeviceContext<
                      paddle::platform::CustomDeviceContext>(p, true)
                      .release();
      int priority = CastPyArg2AttrInt(PyTuple_GET_ITEM(args, 1), 1);
      auto priority_phi = phi::stream::Stream::Priority(2);
      if (priority == -1) {
        priority_phi = phi::stream::Stream::Priority(1);
      }
      auto* custom_ctx =
          dynamic_cast<paddle::platform::CustomDeviceContext*>(v->stream);
      auto stream = std::make_shared<phi::stream::Stream>();
      stream->Init(p, priority_phi);
      custom_ctx->SetStream(stream);
#else
      PADDLE_THROW(platform::errors::Unimplemented(
          "CustomPlace is not supported. Please re-compile with "
          "WITH_CUSTOM_DEVICE "
          "option."));
#endif
    } else {
      PADDLE_THROW(platform::errors::Unimplemented(
          "Not support %s place rigth now.", p));
    }
  }
  return obj;
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static void StreamBaseDealloc(StreamBaseObject* self) {
  if (self->owned_stream_) {
    if (platform::is_gpu_place(self->stream->GetPlace())) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      auto* gpu_ctx = dynamic_cast<phi::GPUContext*>(self->stream);
      delete (gpu_ctx->cuda_stream());
#else
      PADDLE_THROW(
          platform::errors::Unimplemented("CUDAPlace is not supported. Please "
                                          "re-compile with WITH_GPU option."));
#endif
    }
    delete self->stream;
  }
  Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

static PyObject* streambase_properties_get_device(StreamBaseObject* self) {
  EAGER_TRY
  return ToPyObject(self->stream->GetPlace());
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* streambase_properties_get_stream(StreamBaseObject* self,
                                                  PyObject*) {
  EAGER_TRY
  return ToPyObject(reinterpret_cast<int64_t>(self->stream));
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* streambase_properties_get_priority(StreamBaseObject* self,
                                                    PyObject*) {
  EAGER_TRY
  auto place = self->stream->GetPlace();
  if (platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    auto* gpu_ctx = dynamic_cast<phi::GPUContext*>(self->stream);
    auto stream = gpu_ctx->cuda_stream()->raw_stream();
    phi::backends::gpu::GPUDeviceGuard guard(place.device);
    int priority = 0;
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamGetPriority(stream, &priority));
    return ToPyObject(priority);
#else
    PADDLE_THROW(
        platform::errors::Unimplemented("CUDAPlace is not supported. Please "
                                        "re-compile with WITH_GPU option."));
#endif
  } else if (platform::is_custom_place(place)) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
    auto* custom_ctx =
        dynamic_cast<paddle::platform::CustomDeviceContext*>(self->stream);
    auto stream = custom_ctx->GetStream();
    int priority = 0;
    // TODO(ronny1996) support stream get priority
    return ToPyObject(priority);
#else
    PADDLE_THROW(platform::errors::Unimplemented(
        "CUDAPlace is not supported. Please "
        "re-compile with WITH_CUSTOM_DEVICE option."));
#endif
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Stream.priority only supporded on GPU."));
  }
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* streambase_method_priority_range(StreamBaseObject* self) {
  EAGER_TRY
  auto place = self->stream->GetPlace();
  if (platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    phi::backends::gpu::GPUDeviceGuard guard(place.device);
    int least_priority, greatest_priority;
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));
    PADDLE_ENFORCE_GE(
        least_priority,
        0,
        platform::errors::External("Unexpected CUDA stream priority range."));
    PADDLE_ENFORCE_LE(
        greatest_priority,
        -1,
        platform::errors::External("Unexpected CUDA stream priority range"));
    auto result = PyTuple_New(2);
    PyTuple_SET_ITEM(result, 0, ToPyObject(least_priority));
    PyTuple_SET_ITEM(result, 0, ToPyObject(greatest_priority));
    return result;
#else
    PADDLE_THROW(
        platform::errors::Unimplemented("CUDAPlace is not supported. Please "
                                        "re-compile with WITH_GPU option."));
#endif
  } else {
    // TODO(ronny1996) support stream get priority range
    PADDLE_THROW(platform::errors::Unimplemented(
        "Stream.priority_range only supporded on GPU."));
  }
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* streambase_method_query(StreamBaseObject* self) {
  EAGER_TRY
  auto place = self->stream->GetPlace();
  if (platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    auto* gpu_ctx = dynamic_cast<phi::GPUContext*>(self->stream);
    return ToPyObject(gpu_ctx->cuda_stream()->Query());
#else
    PADDLE_THROW(
        platform::errors::Unimplemented("CUDAPlace is not supported. Please "
                                        "re-compile with WITH_GPU option."));
#endif
  } else if (platform::is_custom_place(place)) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
    auto* custom_ctx =
        dynamic_cast<paddle::platform::CustomDeviceContext*>(self->stream);
    auto stream = custom_ctx->GetStream();
    return ToPyObject(stream->Query());
#else
    PADDLE_THROW(platform::errors::Unimplemented(
        "CUDAPlace is not supported. Please "
        "re-compile with WITH_CUSTOM_DEVICE option."));
#endif
  } else {
    PADDLE_THROW(
        platform::errors::Unimplemented("Stream.query only supporded on GPU."));
  }
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* streambase_method_synchronize(StreamBaseObject* self) {
  EAGER_TRY
  auto place = self->stream->GetPlace();
  if (platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    eager_gil_scoped_release guard;
    auto* gpu_ctx = dynamic_cast<phi::GPUContext*>(self->stream);
    gpu_ctx->cuda_stream()->Synchronize();
    RETURN_PY_NONE
#else
    PADDLE_THROW(
        platform::errors::Unimplemented("CUDAPlace is not supported. Please "
                                        "re-compile with WITH_GPU option."));
#endif
  } else if (platform::is_custom_place(place)) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
    auto* custom_ctx =
        dynamic_cast<paddle::platform::CustomDeviceContext*>(self->stream);
    auto stream = custom_ctx->GetStream();
    stream->Synchronize();
    RETURN_PY_NONE
#else
    PADDLE_THROW(platform::errors::Unimplemented(
        "CUDAPlace is not supported. Please "
        "re-compile with WITH_CUSTOM_DEVICE option."));
#endif
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Stream.synchronize only supporded on GPU."));
  }
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* streambase_method___eq__(StreamBaseObject* self,
                                          StreamBaseObject* other) {
  EAGER_TRY
  return ToPyObject(self->stream == other->stream);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyMethodDef streambase_methods[] = {
    {"query",
     (PyCFunction)(void (*)(void))streambase_method_query,
     METH_NOARGS,
     NULL},
    {"synchronize",
     (PyCFunction)(void (*)(void))streambase_method_synchronize,
     METH_NOARGS,
     NULL},
    {"priority_range",
     (PyCFunction)(void (*)(void))streambase_method_priority_range,
     METH_STATIC | METH_NOARGS,
     NULL},
    {"__eq__",
     (PyCFunction)(void (*)(void))streambase_method___eq__,
     METH_O,
     NULL},
    {NULL, NULL, 0, NULL}};

struct PyGetSetDef streambase_properties[] {
  {"stream",
   (getter)streambase_properties_get_stream,
   nullptr,
   nullptr,
   nullptr},
      {"priority",
       (getter)streambase_properties_get_priority,
       nullptr,
       nullptr,
       nullptr},
  {
    nullptr, nullptr, nullptr, nullptr, nullptr
  }
};

void BindEagerDeviceStream(PyObject* module) {
  auto heap_type = reinterpret_cast<PyHeapTypeObject*>(
      PyType_Type.tp_alloc(&PyType_Type, 0));
  heap_type->ht_name = ToPyObject("StreamBase");
  heap_type->ht_qualname = ToPyObject("StreamBase");
  auto type = &heap_type->ht_type;
  type->tp_name = "StreamBase";
  type->tp_basicsize = sizeof(StreamBaseObject);
  type->tp_dealloc = (destructor)StreamBaseDealloc;
  type->tp_methods = streambase_methods;
  type->tp_getset = streambase_properties;
  type->tp_new = (newfunc)StreamBaseNew;
  Py_INCREF(&PyBaseObject_Type);
  type->tp_base = reinterpret_cast<PyTypeObject*>(&PyBaseObject_Type);
  type->tp_flags |=
      Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HEAPTYPE;
#if PY_VERSION_HEX >= 0x03050000
  type->tp_as_async = &heap_type->as_async;
#endif
  p_streambase_type = type;

  if (PyType_Ready(type) < 0) {
    PADDLE_THROW(platform::errors::Fatal(
        "Init Paddle error in BindEagerDeviceStream(PyType_Ready)."));
    return;
  }

  Py_INCREF(type);
  if (PyModule_AddObject(
          module, "StreamBase", reinterpret_cast<PyObject*>(type)) < 0) {
    Py_DECREF(type);
    Py_DECREF(module);
    PADDLE_THROW(platform::errors::Fatal(
        "Init Paddle error in BindEagerDeviceStream(PyModule_AddObject)."));
    return;
  }
}

}  // namespace pybind
}  // namespace paddle
