/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/pybind/executor_lite.h"
#include <pybind11/stl.h>
#include <memory>
#include <vector>
#include "paddle/fluid/lite/api/cxx_api.h"
#include "paddle/fluid/lite/api/paddle_use_passes.h"
#include "paddle/fluid/lite/core/hvy_tensor.h"
#include "paddle/fluid/lite/core/scope.h"
#include "pybind11/pybind11.h"

namespace lt = paddle::lite;
namespace py = pybind11;

namespace paddle {
namespace pybind {

void BindTensor(pybind11::module* m) {
  pybind11::class_<lt::TensorHvy>(*m, "Tensor")
      .def(pybind11::init<>())
      .def("raw_tensor", [](lt::TensorHvy& self) { return self.raw_tensor(); })
      .def("share_data_with",
           [](lt::TensorHvy& self, const framework::Tensor& other) {
             self.ShareDataWith(other);
           });
}

void BindVariable(pybind11::module* m) {
  pybind11::class_<lt::Variable>(*m, "Variable")
      .def("get_mutable_tensor",
           [](lt::Variable& self) { return self.GetMutable<lt::Tensor>(); })
      .def("get_mutable_fetch_list",
           [](lt::Variable& self) -> paddle::lite::FeedFetchList* {
             return self.GetMutable<paddle::lite::FeedFetchList>();
           },
           py::return_value_policy::reference);
}

void BindScope(pybind11::module* m) {
  py::class_<lt::Scope, std::shared_ptr<lt::Scope>>(*m, "Scope")
      .def(pybind11::init<>())
      .def("new_scope",
           [](lt::Scope& self) -> lt::Scope* { return &self.NewScope(); },
           py::return_value_policy::reference)
      .def("var", &lt::Scope::Var, pybind11::return_value_policy::reference)
      .def("find_var", &lt::Scope::FindVar,
           pybind11::return_value_policy::reference)
      .def("find_local_var", &lt::Scope::FindLocalVar,
           pybind11::return_value_policy::reference)
      .def("parent", &lt::Scope::parent,
           pybind11::return_value_policy::reference)
      .def("local_var_names", &lt::Scope::LocalVarNames,
           pybind11::return_value_policy::reference);
}

void BindExecutorLite(pybind11::module* m) {
  py::class_<lt::Predictor>(*m, "Predictor")
      .def(pybind11::init<>())
      .def("__init__",
           [](lt::Predictor& self,
              const std::shared_ptr<lt::Scope>& root_scope) {
             new (&self) lt::Predictor(root_scope);
           })
      .def("get_input", &lt::Predictor::GetInput,
           pybind11::return_value_policy::reference)
      .def("get_output", &lt::Predictor::GetOutput,
           pybind11::return_value_policy::reference)
      .def("run", [](lt::Predictor& self) { self.Run(); })
      .def("run", [](lt::Predictor& self,
                     const std::vector<framework::Tensor>& tensors) {
        self.Run(tensors);
      });
}

void BindEnums(pybind11::module* m) {
  py::enum_<lt::TargetType>(*m, "TargetType", py::arithmetic(),
                            "TargetType enum")
      .value("kUnk", lt::TargetType::kUnk)
      .value("kHost", lt::TargetType::kHost)
      .value("kX86", lt::TargetType::kX86)
      .value("kCUDA", lt::TargetType::kCUDA)
      .value("kARM", lt::TargetType::kARM)
      .value("kAny", lt::TargetType::kAny)
      .value("NUM", lt::TargetType::NUM);

  py::enum_<lt::PrecisionType>(*m, "PrecisionType", py::arithmetic(),
                               "PrecisionType enum")
      .value("kUnk", lt::PrecisionType::kUnk)
      .value("kFloat", lt::PrecisionType::kFloat)
      .value("kInt8", lt::PrecisionType::kInt8)
      .value("kAny", lt::PrecisionType::kAny)
      .value("NUM", lt::PrecisionType::NUM);

  py::enum_<lt::DataLayoutType>(*m, "DataLayoutType", py::arithmetic(),
                                "DataLayoutType enum")
      .value("kUnk", lt::DataLayoutType::kUnk)
      .value("kNCHW", lt::DataLayoutType::kNCHW)
      .value("kAny", lt::DataLayoutType::kAny)
      .value("NUM", lt::DataLayoutType::NUM);
}

void BindPlace(pybind11::module* m) {
  pybind11::class_<lt::Place, std::shared_ptr<lt::Place>>(*m, "Place")
      .def(pybind11::init<>())
      .def("__init__",
           [](lt::Place& self, lt::TargetType target,
              lt::PrecisionType precision, lt::DataLayoutType layout,
              int16_t device) {
             new (&self) lt::Place(target, precision, layout, device);
           })
      .def("is_valid", &lt::Place::is_valid,
           pybind11::return_value_policy::reference);
}

void BindCXXTrainer(pybind11::module* m) {
  pybind11::class_<lt::CXXTrainer, std::shared_ptr<lt::CXXTrainer>>(
      *m, "CXXTrainer")
      .def(
          "__init__",
          [](lt::CXXTrainer& self, const std::shared_ptr<lt::Scope>& root_scope,
             const lt::Place& preferred_place,
             const std::vector<lt::Place>& valid_places) {
            new (&self)
                lt::CXXTrainer(root_scope, preferred_place, valid_places);
          })
      .def("build_main_program_executor",
           [](lt::CXXTrainer& self,
              framework::ProgramDesc& desc) -> lt::Predictor& {
             return self.BuildMainProgramExecutor(desc);
           },
           pybind11::return_value_policy::reference)
      .def("run_startup_program",
           [](lt::CXXTrainer& self, framework::ProgramDesc& desc) {
             return self.RunStartupProgram(desc);
           });
}

void BindLite(pybind11::module* m) {
  BindTensor(m);
  BindVariable(m);
  BindScope(m);
  BindExecutorLite(m);
  BindEnums(m);
  BindPlace(m);
  BindCXXTrainer(m);
}

}  // namespace pybind
}  // namespace paddle

// USE_LITE_OP(mul);
USE_LITE_OP(elementwise_sub);
USE_LITE_OP(uniform_random);
USE_LITE_OP(feed);
USE_LITE_OP(fetch);
USE_LITE_OP(fill_constant);
USE_LITE_OP(mul);
USE_LITE_OP(mul_grad);
USE_LITE_OP(mean);
USE_LITE_OP(square);
USE_LITE_OP(sgd);

USE_LITE_KERNEL(feed, kHost, kAny, kAny, def);
USE_LITE_KERNEL(fetch, kHost, kAny, kAny, def);

#ifdef LITE_WITH_X86
USE_LITE_KERNEL(uniform_random, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(fill_constant, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(mul, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(mul_grad, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(square, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(mean, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(sgd, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(elementwise_sub, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(elementwise_sub_grad, kX86, kFloat, kNCHW, def);
#endif
