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

#include "paddle/cinn/backends/cuda_util.h"
#include "paddle/cinn/common/cinn_value.h"
#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/cinn/hlir/framework/op_strategy.h"
#include "paddle/cinn/hlir/op/use_ops.h"
#include "paddle/cinn/pybind/bind.h"
#include "paddle/cinn/runtime/flags.h"

#include "paddle/cinn/runtime/backend_api.h"
using cinn::runtime::BackendAPI;

namespace cinn::pybind {

namespace py = pybind11;
using namespace cinn::hlir::framework;  // NOLINT
void BindFramework(pybind11::module *m) {
  py::class_<Operator>(*m, "Operator")
      .def("get_op_attrs", [](const std::string &key) {
        return Operator::GetAttrs<StrategyFunction>(key);
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
                               ::common::errors::InvalidArgument(
                                   "Didn't find value with key [%d].",
                                   self.attr_store.count(key)));
             return self.attr_store[key];
           })
      .def("__str__", [](NodeAttr &self) { return utils::GetStreamCnt(self); });
}
}  // namespace cinn::pybind
