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
              const common::Target &target) {
             const Operator *op_ptr = Operator::Get(key);
             auto impl = OpStrategy::SelectImpl(
                 self[op_ptr](attrs, inputs, out_types, output_shapes, target));
             std::vector<common::CINNValue> temp_inputs;
             std::vector<ir::Tensor> res;
             for (auto &tensor : inputs) {
               res.push_back(tensor);
               temp_inputs.push_back(common::CINNValue(tensor));
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
                     common::CINNValuePack{temp_inputs},
                     res,
                     input_output_names,
                     key,
                     target);
             CHECK_EQ(funcs.size(), 1U);
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
             CHECK_EQ(self.attr_store.count(key), 1)
                 << "Didn't find value with key [" << key << "].";
             return self.attr_store[key];
           })
      .def("__str__", [](NodeAttr &self) { return utils::GetStreamCnt(self); });

  py::class_<Scope, std::shared_ptr<Scope>>(*m, "Scope")
      .def(py::init<>())  //
      .def("get_tensor",
           [](Scope &self, const std::string &name, const Target &target) {
             auto t = self.GetTensor(name);
             py::dtype dt(common::Type2Str(t->type()));
             py::array::ShapeContainer shape(t->shape().data().begin(),
                                             t->shape().data().end());
             py::array array(std::move(dt), std::move(shape));
             auto *mutable_data = array.mutable_data();
             if (target.arch == Target::Arch::X86) {
               std::memcpy(mutable_data,
                           t->data<void>(),
                           t->shape().numel() * t->type().bytes());
             } else if (target.arch == Target::Arch::NVGPU) {
#ifdef CINN_WITH_CUDA
               CUDA_CALL(cudaMemcpy(
                   mutable_data,
                   reinterpret_cast<void *>(t->mutable_data(target, t->type())),
                   t->shape().numel() * t->type().bytes(),
                   cudaMemcpyDeviceToHost));
#else
    LOG(FATAL) <<"To use CUDA backends, you need to set WITH_CUDA ON!";
#endif
             } else {
               CINN_NOT_IMPLEMENTED
             }
             return array;
           })
      .def("var_names", &Scope::var_names);

  py::class_<common::Shared<hlir::framework::_Tensor_>>(*m, "SharedTensor");
  py::class_<Tensor, common::Shared<hlir::framework::_Tensor_>>(*m, "Tensor")
      .def(py::init<>())
      .def("shape",
           [](hlir::framework::Tensor &self) { return self->shape().data(); })
      .def("set_type",
           [](hlir::framework::Tensor &self, Type type) {
             self->set_type(type);
           })
      .def(
          "numpy",
          [](hlir::framework::Tensor &self, const common::Target &target) {
            std::string type_str = common::Type2Str(self->type());
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
            } else if (target.arch == Target::Arch::NVGPU) {
#ifdef CINN_WITH_CUDA
              CUDA_CALL(cudaMemcpy(array_data,
                                   self->data<void>(),
                                   self->shape().numel() * self->type().bytes(),
                                   cudaMemcpyDeviceToHost));
#else
    LOG(FATAL) <<"To use CUDA backends, you need to set WITH_CUDA ON!";
#endif
            } else {
              CINN_NOT_IMPLEMENTED
            }
            return array;
          })
      .def(
          "from_numpy",
          [](hlir::framework::Tensor &self,
             py::array array,
             const common::Target &target) {
            CHECK(array.dtype().is(py::dtype(common::Type2Str(self->type()))))
                << "currently only support float32 data type as input";
            hlir::framework::shape_t shape;
            std::copy_n(array.shape(), array.ndim(), std::back_inserter(shape));
            CHECK_EQ(
                std::accumulate(shape.begin(),
                                shape.end(),
                                1,
                                [](int32_t a, int32_t b) { return a * b; }),
                self->shape().numel());
            auto *data = self->mutable_data(target, self->type());
            if (target.arch == Target::Arch::X86) {
              std::memcpy(data,
                          array.data(),
                          self->shape().numel() * self->type().bytes());
            } else if (target.arch == Target::Arch::NVGPU) {
#ifdef CINN_WITH_CUDA
              CUDA_CALL(cudaMemcpy(reinterpret_cast<void *>(data),
                                   reinterpret_cast<const void *>(array.data()),
                                   self->shape().numel() * self->type().bytes(),
                                   cudaMemcpyHostToDevice));
#else
    LOG(FATAL) <<"To use CUDA backends, you need to set WITH_CUDA ON!";
#endif
            } else {
              CINN_NOT_IMPLEMENTED
            }
          });
}
}  // namespace cinn::pybind
