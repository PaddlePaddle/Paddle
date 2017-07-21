/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <Python.h>
#include <paddle/framework/op_registry.h>
#include <paddle/framework/operator.h>
#include <paddle/framework/scope.h>
#include <paddle/pybind/tensor_bind.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <fstream>
#include <vector>

namespace py = pybind11;
namespace pd = paddle::framework;

USE_OP(add_two);
USE_OP(onehot_cross_entropy);
USE_OP_WITHOUT_KERNEL(fc);
USE_OP(sgd);
USE_OP(mul);
USE_OP(sigmoid);
USE_OP(softmax);
USE_OP(rowwise_add);

PYBIND11_PLUGIN(core) {
  py::module m("core", "C++ core of Paddle Paddle");

  py::class_<pd::Tensor>(m, "Tensor", py::buffer_protocol())
      .def_buffer([](pd::Tensor& self) -> py::buffer_info {
        return paddle::pybind::CastToPyBuffer(self);
      })
      .def("get_dims",
           [](const pd::Tensor& self) { return pd::vectorize(self.dims()); })
      .def("set_dims",
           [](pd::Tensor& self, const std::vector<int>& dim) {
             self.Resize(pd::make_ddim(dim));
           })
      .def("alloc_float",
           [](pd::Tensor& self) {
             self.mutable_data<float>(paddle::platform::CPUPlace());
           })
      .def("alloc_int",
           [](pd::Tensor& self) {
             self.mutable_data<int>(paddle::platform::CPUPlace());
           })
      .def("set", paddle::pybind::PyTensorSetFromArray<float>)
      .def("set", paddle::pybind::PyTensorSetFromArray<int>)
      .def("shape",
           [](pd::Tensor& self) { return pd::vectorize(self.dims()); });

  py::class_<pd::Variable>(m, "Variable", R"DOC(Variable Class.

All parameter, weight, gradient are variables in Paddle.
)DOC")
      .def("is_int", [](const pd::Variable& var) { return var.IsType<int>(); })
      .def("set_int",
           [](pd::Variable& var, int val) -> void {
             *var.GetMutable<int>() = val;
           })
      .def("get_int",
           [](const pd::Variable& var) -> int { return var.Get<int>(); })
      .def("get_tensor",
           [](pd::Variable& self) -> pd::Tensor* {
             return self.GetMutable<pd::Tensor>();
           },
           py::return_value_policy::reference);

  py::class_<pd::Scope, std::shared_ptr<pd::Scope>>(m, "Scope")
      .def(py::init<const std::shared_ptr<pd::Scope>&>())
      .def("get_var",
           &pd::Scope::GetVariable,
           py::return_value_policy::reference)
      .def("create_var",
           &pd::Scope::CreateVariable,
           py::return_value_policy::reference);

  //! @note: Be careful! PyBind will return std::string as an unicode, not
  //! Python str. If you want a str object, you should cast them in Python.
  m.def("get_all_op_protos", []() -> std::vector<py::bytes> {
    auto& protos = pd::OpRegistry::protos();
    std::vector<py::bytes> ret_values;
    for (auto it = protos.begin(); it != protos.end(); ++it) {
      PADDLE_ENFORCE(it->second.IsInitialized(),
                     "OpProto must all be initialized");
      std::string str;
      PADDLE_ENFORCE(it->second.SerializeToString(&str),
                     "Serialize OpProto Error. This could be a bug of Paddle.");
      ret_values.push_back(py::bytes(str));
    }
    return ret_values;
  });
  m.def_submodule(
       "var_names",
       "The module will return special predefined variable name in Paddle")
      .def("empty", pd::OperatorBase::EMPTY_VAR_NAME)
      .def("temp", pd::OperatorBase::TMP_VAR_NAME);

  py::class_<paddle::platform::DeviceContext>(m, "DeviceContext")
      .def_static("cpu_context", []() -> paddle::platform::DeviceContext* {
        return new paddle::platform::CPUDeviceContext();
      });

  py::class_<pd::OperatorBase, pd::OperatorPtr>(m, "Operator")
      .def("__str__", &pd::OperatorBase::DebugString)
      .def_static("create",
                  [](py::bytes protobin) {
                    pd::OpDesc desc;
                    PADDLE_ENFORCE(desc.ParsePartialFromString(protobin),
                                   "Cannot parse user input to OpDesc");
                    PADDLE_ENFORCE(desc.IsInitialized(),
                                   "User OpDesc is not initialized, reason %s",
                                   desc.InitializationErrorString());
                    return pd::OpRegistry::CreateOp(desc);
                  })
      .def("infer_shape", &pd::OperatorBase::InferShape)
      .def("run", &pd::OperatorBase::Run)
      .def("outputs", [](const pd::OperatorPtr& op) { return op->outputs_; });

  return m.ptr();
}
