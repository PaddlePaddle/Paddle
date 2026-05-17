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

#include "paddle/fluid/pybind/jit.h"
#include "glog/logging.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/jit/function.h"
#include "paddle/fluid/jit/function_schema.h"
#include "paddle/fluid/jit/layer.h"
#include "paddle/fluid/jit/serializer.h"
#include "paddle/fluid/pybind/sot/eval_frame.h"
#include "paddle/fluid/pybind/sot/eval_frame_tools.h"
#include "paddle/fluid/pybind/sot/frame_proxy.h"
#include "paddle/fluid/pybind/sot/guards.h"
#include "paddle/fluid/pybind/sot/macros.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"
#include "paddle/utils/pybind.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"

namespace py = pybind11;

namespace paddle::pybind {

PyTypeObject *g_jit_function_pytype = nullptr;
using Variable = paddle::framework::Variable;

void BindJit(pybind11::module *m) {
  py::class_<jit::Layer>(*m, "Layer", R"DOC(Layer Class.)DOC")
      .def("function_names", &jit::Layer::FunctionNames)
      .def("function", &jit::Layer::Function)
      .def("function_info", &jit::Layer::FunctionInfo);

  py::class_<jit::Function, std::shared_ptr<jit::Function>> function(
      *m, "Function", R"DOC(Function Class.)DOC");
  g_jit_function_pytype = reinterpret_cast<PyTypeObject *>(function.ptr());

  py::class_<jit::BaseFunctionInfo, std::shared_ptr<jit::BaseFunctionInfo>>(
      *m, "FunctionInfo", R"DOC(BaseFunctionInfo Class.)DOC")
      .def("name", &jit::BaseFunctionInfo::FunctionName)
      .def("input_names", &jit::BaseFunctionInfo::InputArgNames)
      .def("output_names", &jit::BaseFunctionInfo::OutputArgNames);

  m->def("Load", [](const std::string &path, const CPUPlace &cpu_place) {
    return paddle::jit::Load(path, cpu_place);
  });

  m->def("Load", [](const std::string &path, const GPUPlace &cuda_place) {
    return paddle::jit::Load(path, cuda_place);
  });
}

void BindGuard(pybind11::module *m) {
#if SOT_IS_SUPPORTED
  py::class_<CompiledGuard, std::shared_ptr<CompiledGuard>>(
      *m, "CompiledGuard", R"DOC(CompiledGuard Class.)DOC")
      .def(py::init<const py::list &>(), py::arg("specs"))
      .def("check", &CompiledGuard::check_pybind, py::arg("frame"))
      .def("stringify", &CompiledGuard::stringify);
  py::class_<CompiledGuardLookup, std::shared_ptr<CompiledGuardLookup>>(
      *m, "CompiledGuardLookup", R"DOC(CompiledGuardLookup Class.)DOC")
      .def(py::init<>())
      .def("add_guard",
           &CompiledGuardLookup::add_guard,
           py::arg("guard"),
           py::arg("cache_index"))
      .def(
          "lookup",
          [](CompiledGuardLookup &self, py::object frame) {
            return self.lookup(reinterpret_cast<FrameProxy *>(frame.ptr()));
          },
          py::arg("frame"))
      .def("stringify", &CompiledGuardLookup::stringify);
#endif
}

void BindSot(pybind11::module *m) {
#if SOT_IS_SUPPORTED
  PyInit__eval_frame();
#if PY_3_11_PLUS
  PyInit__frame_proxy();
#endif
  m->def(
      "set_eval_frame",
      [](const py::object &py_func) {
        VLOG(5) << "start call set_eval_frame_py.";
        auto ret = set_eval_frame_py(py_func.ptr());
        auto obj = py::reinterpret_borrow<py::object>(ret);
        return obj;
      },
      py::arg("callback"));

  m->def("has_custom_getattro", [](py::object obj) {
    PyObject *py_obj = obj.ptr();

    if (!PyType_Check(py_obj)) {
      PADDLE_THROW(common::errors::InvalidArgument(
          "The input object should be a type object, but got %s.",
          py::str(py_obj).cast<std::string>()));
    }
    PyTypeObject *type = reinterpret_cast<PyTypeObject *>(py_obj);

    return type->tp_getattro != PyObject_GenericGetAttr;
  });

  m->def(
      "sot_setup_codes_with_graph",
      [](const py::object &py_codes) {
        auto ret = setup_codes_with_graph(py_codes.ptr());
        auto obj = py::reinterpret_borrow<py::object>(ret);
        return obj;
      },
      py::arg("py_codes"));

  m->def(
      "sot_set_with_graph",
      [](const py::object &py_codes) {
        auto ret = set_with_graph(py_codes.ptr());
        auto obj = py::reinterpret_borrow<py::object>(ret);
        return obj;
      },
      py::arg("py_codes"));

  m->def(
      "eval_frame_no_skip_codes",
      [](const py::object &py_codes) {
        auto ret = no_skip_codes(py_codes.ptr());
        auto obj = py::reinterpret_borrow<py::object>(ret);
        return obj;
      },
      py::arg("py_codes"));

  m->def(
      "eval_frame_skip_file_prefix",
      [](const py::object &py_codes) {
        auto ret = skip_file_prefix(py_codes.ptr());
        auto obj = py::reinterpret_borrow<py::object>(ret);
        return obj;
      },
      py::arg("py_codes"));
  BindGuard(m);
#endif
}

}  // namespace paddle::pybind
