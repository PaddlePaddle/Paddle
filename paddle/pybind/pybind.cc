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
#include <fstream>
#include <vector>

#include "paddle/framework/net.h"
#include "paddle/framework/op_registry.h"
#include "paddle/framework/operator.h"
#include "paddle/framework/scope.h"
#include "paddle/pybind/tensor_bind.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

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
USE_OP_WITHOUT_KERNEL(recurrent_op);

template <typename ClassType>
void ExposeOperator(ClassType& m) {
  m.def("infer_shape", &ClassType::type::InferShape)
      .def("run", &ClassType::type::Run)
      .def("outputs",
           [](const typename ClassType::type& op) -> std::vector<std::string> {
             return op.outputs_;
           })
      .def("__str__", &ClassType::type::DebugString);
}

static size_t UniqueIntegerGenerator() {
  static std::atomic<size_t> generator;
  return generator.fetch_add(1);
}

PYBIND11_PLUGIN(core) {
  py::module m("core", "C++ core of PaddlePaddle");

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
           py::return_value_policy::reference)
      .def("get_net",
           [](pd::Variable& self) -> pd::NetOp* {
             return self.GetMutable<pd::NetOp>();
           },
           py::return_value_policy::reference);

  py::class_<pd::Scope>(m, "Scope", "")
      .def("new_var",
           [](pd::Scope& self, const std::string& name) -> pd::Variable* {
             return self.NewVar(name);
           },
           py::return_value_policy::reference)
      .def("find_var", &pd::Scope::FindVar, py::return_value_policy::reference)
      .def(py::init<>())
      .def("new_scope",
           [](pd::Scope& self) -> pd::Scope* { return &self.NewScope(); },
           py::return_value_policy::reference)
      .def("drop_kids", &pd::Scope::DropKids);

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

  py::class_<pd::OperatorBase, std::shared_ptr<pd::OperatorBase>> operator_base(
      m, "Operator");

  operator_base.def_static("create", [](py::bytes protobin) {
    pd::OpDesc desc;
    PADDLE_ENFORCE(desc.ParsePartialFromString(protobin),
                   "Cannot parse user input to OpDesc");
    PADDLE_ENFORCE(desc.IsInitialized(),
                   "User OpDesc is not initialized, reason %s",
                   desc.InitializationErrorString());
    return pd::OpRegistry::CreateOp(desc);
  });
  ExposeOperator(operator_base);

  py::class_<pd::NetOp, std::shared_ptr<pd::NetOp>> net(m, "Net");

  net.def_static("create",
                 []() -> std::shared_ptr<pd::NetOp> {
                   auto retv = std::make_shared<pd::NetOp>();
                   retv->type_ = "plain_net";
                   return retv;
                 })
      .def("add_op", &pd::NetOp::AddOp)
      .def("add_op",
           [](pd::NetOp& self, const std::shared_ptr<pd::NetOp>& net) -> void {
             self.AddOp(std::static_pointer_cast<pd::OperatorBase>(net));
           })
      .def("complete_add_op", &pd::NetOp::CompleteAddOp)
      .def("complete_add_op",
           [](std::shared_ptr<pd::NetOp>& self) { self->CompleteAddOp(); });
  ExposeOperator(net);

  m.def("unique_integer", UniqueIntegerGenerator);

  return m.ptr();
}
