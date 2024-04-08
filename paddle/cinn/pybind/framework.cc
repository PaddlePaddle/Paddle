// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>

#include "paddle/cinn/common/cinn_value.h"
#include "paddle/cinn/frontend/interpreter.h"
#include "paddle/cinn/hlir/framework/graph_compiler.h"
#include "paddle/cinn/hlir/framework/instruction.h"
#include "paddle/cinn/hlir/framework/node.h"
#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/cinn/hlir/framework/op_strategy.h"
#include "paddle/cinn/hlir/framework/scope.h"
#include "paddle/cinn/hlir/op/use_ops.h"
#include "paddle/cinn/pybind/bind.h"
#include "paddle/cinn/runtime/flags.h"

namespace cinn::pybind {

namespace py = pybind11;
using namespace cinn::hlir::framework;  // NOLINT
void BindFramework(pybind11::module *m) {
  py::class_<Operator>(*m, "Operator")
      .def("get_op_attrs",
           [](const std::string &key) {
             return Operator::GetAttrs<StrategyFunction>(key);
           })
      .def("get_op_shape_attrs", [](const std::string &key) {
        return Operator::GetAttrs<InferShapeFunction>(key);
      });

  py::class_<OpValueType<StrategyFunction>>(*m, "OpValueType")
      .def("apply_strategy",
           [](OpValueType<StrategyFunction> &self,
              const std::string &key,
              const NodeAttr &attrs,
              const std::vector<ir::Tensor> &inputs,
              const std::vector<Type> &out_types,
              const std::vector<std::vector<int>> &output_shapes,
              const cinn::common::Target &target) {
             const Operator *op_ptr = Operator::Get(key);
             auto impl = OpStrategy::SelectImpl(
                 self[op_ptr](attrs, inputs, out_types, output_shapes, target));
             std::vector<cinn::common::CINNValue> temp_inputs;
             std::vector<ir::Tensor> res;
             for (auto &tensor : inputs) {
               res.push_back(tensor);
               temp_inputs.push_back(cinn::common::CINNValue(tensor));
             }

             ir::LoweredFunc func;
             std::string output_name = "out";
             temp_inputs.emplace_back(output_name);
             std::vector<std::string> input_output_names;
             for (const auto &input : inputs) {
               input_output_names.push_back(input->name);
             }
             input_output_names.push_back(output_name);
             std::vector<ir::LoweredFunc> funcs =
                 hlir::framework::GetFuncFromImpl(
                     impl,
                     cinn::common::CINNValuePack{temp_inputs},
                     res,
                     input_output_names,
                     key,
                     target);
             PADDLE_ENFORCE_EQ(funcs.size(),
                               1U,
                               phi::errors::InvalidArgument(
                                   "The size of funcs is incorrect."
                                   "Expected size is 1, but receive %d.",
                                   funcs.size()));
             func = funcs[0];
             return func;
           });

  py::class_<OpValueType<InferShapeFunction>>(*m, "OpValueType1")
      .def("infer_shape",
           [](OpValueType<InferShapeFunction> &self,
              const std::string &key,
              const std::vector<std::vector<int>> &input_shapes,
              const AttrMapType &attrs) {
             const Operator *op_ptr = Operator::Get(key);
             auto shapes = self[op_ptr](input_shapes, attrs);
             return shapes;
           });

  py::class_<NodeAttr>(*m, "NodeAttr")
      .def(py::init<>())
      .def_readwrite("attr_store", &NodeAttr::attr_store)
      .def("set_attr",
           [](NodeAttr &self, const std::string &key, NodeAttr::attr_t value) {
             self.attr_store[key] = value;
           })
      .def("get_attr",
           [](NodeAttr &self, const std::string &key) {
             PADDLE_ENFORCE_EQ(self.attr_store.count(key),
                               1,
                               phi::errors::InvalidArgument(
                                   "Didn't find value with key [%d].",
                                   self.attr_store.count(key)));
             return self.attr_store[key];
           })
      .def("__str__", [](NodeAttr &self) { return utils::GetStreamCnt(self); });

  py::class_<Scope, std::shared_ptr<Scope>>(*m, "Scope")
      .def(py::init<>())  //
      .def("get_tensor",
           [](Scope &self, const std::string &name, const Target &target) {
             auto t = self.GetTensor(name);
             py::dtype dt(cinn::common::Type2Str(t->type()));
             py::array::ShapeContainer shape(t->shape().data().begin(),
                                             t->shape().data().end());
             py::array array(std::move(dt), std::move(shape));
             auto *mutable_data = array.mutable_data();
             if (target.arch == Target::Arch::X86) {
               std::memcpy(mutable_data,
                           t->data<void>(),
                           t->shape().numel() * t->type().bytes());
             } else if (target.arch_is_gpu()) {
               using cinn::runtime::BackendAPI;
               BackendAPI::get_backend(target)->memcpy(
                   mutable_data,
                   reinterpret_cast<void *>(t->mutable_data(target, t->type())),
                   t->shape().numel() * t->type().bytes(),
                   BackendAPI::MemcpyType::DeviceToHost);
             } else {
               CINN_NOT_IMPLEMENTED
             }
             return array;
           })
      .def("var_names", &Scope::var_names);

  py::class_<cinn::common::Shared<hlir::framework::_Tensor_>>(*m,
                                                              "SharedTensor");
  py::class_<Tensor, cinn::common::Shared<hlir::framework::_Tensor_>>(*m,
                                                                      "Tensor")
      .def(py::init<>())
      .def("shape",
           [](hlir::framework::Tensor &self) { return self->shape().data(); })
      .def("set_type",
           [](hlir::framework::Tensor &self, Type type) {
             self->set_type(type);
           })
      .def("numpy",
           [](hlir::framework::Tensor &self,
              const cinn::common::Target &target) {
             std::string type_str = cinn::common::Type2Str(self->type());
             if (type_str == "bfloat16") {
               type_str = "uint16";
             }
             py::dtype dt(type_str);
             py::array::ShapeContainer shape(self->shape().data().begin(),
                                             self->shape().data().end());
             py::array array(std::move(dt), std::move(shape));
             void *array_data = array.mutable_data();
             if (target.arch == Target::Arch::X86) {
               std::memcpy(array_data,
                           self->data<void>(),
                           self->shape().numel() * self->type().bytes());
             } else if (target.arch_is_gpu()) {
               using cinn::runtime::BackendAPI;
               BackendAPI::get_backend(target)->memcpy(
                   array_data,
                   self->data<void>(),
                   self->shape().numel() * self->type().bytes(),
                   BackendAPI::MemcpyType::DeviceToHost);
             } else {
               CINN_NOT_IMPLEMENTED
             }
             return array;
           })
      .def(
          "from_numpy",
          [](hlir::framework::Tensor &self,
             py::array array,
             const cinn::common::Target &target) {
            CHECK(array.dtype().is(
                py::dtype(cinn::common::Type2Str(self->type()))))
                << "currently only support float32 data type as input";
            hlir::framework::shape_t shape;
            std::copy_n(array.shape(), array.ndim(), std::back_inserter(shape));
            PADDLE_ENFORCE_EQ(
                std::accumulate(shape.begin(),
                                shape.end(),
                                1,
                                [](int32_t a, int32_t b) { return a * b; }),
                self->shape().numel(),
                phi::errors::InvalidArgument(
                    "The product of all elements in the shape container and "
                    "shape numel is not equal,"
                    "where the product of all elements in the shape "
                    "container:%d but shape numel:%d.",
                    std::accumulate(shape.begin(),
                                    shape.end(),
                                    1,
                                    [](int32_t a, int32_t b) { return a * b; }),
                    self->shape().numel()));
            auto *data = self->mutable_data(target, self->type());
            if (target.arch == Target::Arch::X86) {
              std::memcpy(data,
                          array.data(),
                          self->shape().numel() * self->type().bytes());
            } else if (target.arch_is_gpu()) {
              using cinn::runtime::BackendAPI;
              BackendAPI::get_backend(target)->memcpy(
                  reinterpret_cast<void *>(data),
                  reinterpret_cast<const void *>(array.data()),
                  self->shape().numel() * self->type().bytes(),
                  BackendAPI::MemcpyType::HostToDevice);
            } else {
              CINN_NOT_IMPLEMENTED
            }
          });

  py::class_<Instruction> instruction(*m, "Instruction");
  instruction
      .def(py::init<const Target &,
                    Scope *,
                    const std::vector<std::string> &,
                    const std::vector<std::string> &,
                    const std::string &>())
      .def("run",
           [](Instruction &self,
              backends::Compiler &compiler,
              const std::string fn_name,
              std::map<std::string, cinn_pod_value_t> &name_to_pod) {
             auto fn_ptr = compiler.Lookup(fn_name);
             self.Finalize();
             self.SetLoweredFunc(fn_ptr);
             self.Run(&name_to_pod);
           });
}
}  // namespace cinn::pybind
