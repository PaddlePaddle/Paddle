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

#include "paddle/framework/backward.h"
#include "paddle/framework/op_registry.h"
#include "paddle/framework/tensor_py.h"
#include "paddle/operators/net_op.h"
#include "paddle/platform/enforce.h"
#include "paddle/platform/place.h"
#include "paddle/string/to_string.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;

USE_OP(add_two);
USE_CPU_ONLY_OP(onehot_cross_entropy);
USE_OP(sgd);
USE_OP(mul);
USE_OP(mean);
USE_OP(sigmoid);
USE_OP(softmax);
USE_OP(rowwise_add);
USE_OP(fill_zeros_like);
USE_OP_ITSELF(recurrent_op);
USE_OP(gaussian_random);
USE_OP(uniform_random);

namespace paddle {
namespace framework {

using Tensor = framework::Tensor;

template <typename ClassType>
void ExposeOperator(ClassType &m) {
  m.def("infer_shape", &ClassType::type::InferShape)
      .def("run", &ClassType::type::Run)
      .def("type",
           [](const typename ClassType::type &op) -> std::string {
             return op.Type();
           })
      .def("outputs",
           [](const typename ClassType::type &op)
               -> std::map<std::string, std::vector<std::string>> {
                 return op.Outputs();
               })
      .def("inputs",
           [](const typename ClassType::type &op) { return op.Inputs(); })
      .def("__str__", &ClassType::type::DebugString)
      .def("no_intermediate_outputs",
           [](const typename ClassType::type &op) {
             return op.OutputVars(false);
           })
      .def("support_gpu", &ClassType::type::SupportGPU);
}

static size_t UniqueIntegerGenerator() {
  static std::atomic<size_t> generator;
  return generator.fetch_add(1);
}

bool IsCompileGPU() {
#ifdef PADDLE_ONLY_CPU
  return false;
#else
  return true;
#endif
}

PYBIND11_PLUGIN(core) {
  py::module m("core", "C++ core of PaddlePaddle");

  py::class_<Tensor>(m, "Tensor", py::buffer_protocol())
      .def_buffer(
          [](Tensor &self) -> py::buffer_info { return CastToPyBuffer(self); })
      .def("get_dims",
           [](const Tensor &self) { return vectorize(self.dims()); })
      .def("set_dims",
           [](Tensor &self, const std::vector<int> &dim) {
             self.Resize(make_ddim(dim));
           })
      .def("alloc_float",
           [](Tensor &self, paddle::platform::GPUPlace &place) {
             self.mutable_data<float>(place);
           })
      .def("alloc_float",
           [](Tensor &self, paddle::platform::CPUPlace &place) {
             self.mutable_data<float>(place);
           })
      .def("alloc_int",
           [](Tensor &self, paddle::platform::CPUPlace &place) {
             self.mutable_data<int>(place);
           })
      .def("alloc_int",
           [](Tensor &self, paddle::platform::GPUPlace &place) {
             self.mutable_data<int>(place);
           })
      .def("set", PyCPUTensorSetFromArray<float>)
      .def("set", PyCPUTensorSetFromArray<int>)
#ifndef PADDLE_ONLY_CPU
      .def("set", PyCUDATensorSetFromArray<float>)
      .def("set", PyCUDATensorSetFromArray<int>)
#endif
      .def("shape", [](Tensor &self) { return vectorize(self.dims()); })
      .def("set_float_element",
           [](Tensor &self, size_t offset, float f) {
             // TODO(yuyang18): Only support GPU now.
             self.data<float>()[offset] = f;
           })
      .def("get_float_element", [](Tensor &self, size_t offset) -> float {
        // TODO(yuyang18): Only support GPU now.
        return self.data<float>()[offset];
      });

  py::class_<Variable>(m, "Variable", R"DOC(Variable Class.

All parameter, weight, gradient are variables in Paddle.
)DOC")
      .def("is_int", [](const Variable &var) { return var.IsType<int>(); })
      .def("set_int",
           [](Variable &var, int val) -> void { *var.GetMutable<int>() = val; })
      .def("get_int", [](const Variable &var) -> int { return var.Get<int>(); })
      .def("get_tensor",
           [](Variable &self) -> Tensor * { return self.GetMutable<Tensor>(); },
           py::return_value_policy::reference)
      .def("get_net",
           [](Variable &self) -> operators::NetOp * {
             return self.GetMutable<operators::NetOp>();
           },
           py::return_value_policy::reference);

  py::class_<Scope>(m, "Scope", "")
      .def("new_var",
           [](Scope &self, const std::string &name) -> Variable * {
             return self.NewVar(name);
           },
           py::return_value_policy::reference)
      .def("find_var", &Scope::FindVar, py::return_value_policy::reference)
      .def(py::init<>())
      .def("new_scope", [](Scope &self) -> Scope * { return &self.NewScope(); },
           py::return_value_policy::reference)
      .def("drop_kids", &Scope::DropKids);

  //! @note: Be careful! PyBind will return std::string as an unicode, not
  //! Python str. If you want a str object, you should cast them in Python.
  m.def("get_all_op_protos", []() -> std::vector<py::bytes> {
    auto &op_info_map = OpRegistry::op_info_map();
    std::vector<py::bytes> ret_values;
    for (auto it = op_info_map.begin(); it != op_info_map.end(); ++it) {
      const OpProto *proto = it->second.proto_;
      if (proto == nullptr) {
        continue;
      }
      PADDLE_ENFORCE(proto->IsInitialized(), "OpProto must all be initialized");
      std::string str;
      PADDLE_ENFORCE(proto->SerializeToString(&str),
                     "Serialize OpProto Error. This could be a bug of Paddle.");
      ret_values.push_back(py::bytes(str));
    }
    return ret_values;
  });
  m.def_submodule(
       "var_names",
       "The module will return special predefined variable name in Paddle")
      .def("empty", []() { return kEmptyVarName; })
      .def("temp", []() { return kTempVarName; });
  // clang-format off
  py::class_<paddle::platform::DeviceContext>(m, "DeviceContext")
      .def_static("create",
                  [](paddle::platform::CPUPlace& place)
                      -> paddle::platform::DeviceContext* {
                    return new paddle::platform::CPUDeviceContext();
                  })
      .def_static("create",
                  [](paddle::platform::GPUPlace& place)
                      -> paddle::platform::DeviceContext* {
#ifdef PADDLE_ONLY_CPU
                    PADDLE_THROW("GPUPlace is not supported in CPU device.");
#else
                    return new paddle::platform::CUDADeviceContext(place);
#endif
                  });
  // clang-format on

  py::class_<platform::GPUPlace>(m, "GPUPlace")
      .def(py::init<int>())
      .def("__str__", string::to_string<const platform::GPUPlace &>);

  py::class_<paddle::platform::CPUPlace>(m, "CPUPlace")
      .def(py::init<>())
      .def("__str__", string::to_string<const platform::CPUPlace &>);

  py::class_<OperatorBase, std::shared_ptr<OperatorBase>> operator_base(
      m, "Operator");

  operator_base.def_static("create", [](py::bytes protobin) {
    OpDesc desc;
    PADDLE_ENFORCE(desc.ParsePartialFromString(protobin),
                   "Cannot parse user input to OpDesc");
    PADDLE_ENFORCE(desc.IsInitialized(),
                   "User OpDesc is not initialized, reason %s",
                   desc.InitializationErrorString());
    return OpRegistry::CreateOp(desc);
  });

  operator_base.def("backward",
                    [](const OperatorBase &forwardOp,
                       const std::unordered_set<std::string> &no_grad_vars) {
                      return Backward(forwardOp, no_grad_vars);
                    });

  ExposeOperator(operator_base);

  py::class_<operators::NetOp, std::shared_ptr<operators::NetOp>> net(m, "Net");

  net.def_static("create",
                 []() -> std::shared_ptr<operators::NetOp> {
                   auto retv = std::make_shared<operators::NetOp>();
                   retv->SetType("plain_net");
                   return retv;
                 })
      .def("add_op", &operators::NetOp::AddOp)
      .def("add_op",
           [](operators::NetOp &self,
              const std::shared_ptr<operators::NetOp> &net) -> void {
             self.AddOp(std::static_pointer_cast<OperatorBase>(net));
           })
      .def("complete_add_op", &operators::NetOp::CompleteAddOp)
      .def("complete_add_op", [](std::shared_ptr<operators::NetOp> &self) {
        self->CompleteAddOp();
      });

  ExposeOperator(net);

  m.def("unique_integer", UniqueIntegerGenerator);

  m.def("is_compile_gpu", IsCompileGPU);

  return m.ptr();
}
}  // namespace framework
}  // namespace paddle
