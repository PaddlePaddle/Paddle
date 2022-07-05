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

#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/platform/place.h"

#include "paddle/fluid/jit/executor_function.h"
#include "paddle/fluid/jit/function_schema.h"
#include "paddle/fluid/jit/layer.h"
#include "paddle/fluid/jit/serializer.h"

namespace py = pybind11;

namespace paddle {
namespace pybind {

using Variable = paddle::framework::Variable;

void BindJit(pybind11::module *m) {
  py::class_<jit::Layer>(*m, "Layer", R"DOC(Layer Class.)DOC")
      .def("function_dict", &jit::Layer::FunctionDict);

  py::class_<jit::ExecutorFunction, std::shared_ptr<jit::ExecutorFunction>>(
      *m, "ExectorFunction", R"DOC(ExectorFunction Class.)DOC")
      .def("__call__",
           [](jit::ExecutorFunction &self,
              const std::vector<std::shared_ptr<imperative::VarBase>>
                  &tensor_inputs) {
             std::vector<Variable> var_inputs;
             for (auto &tensor : tensor_inputs) {
               var_inputs.emplace_back(tensor->Var());
             }
             auto var_outputs = self(var_inputs);

             std::vector<std::shared_ptr<imperative::VarBase>> tensor_outputs;
             auto output_names = self.Info()->OutputArgNames();
             for (size_t i = 0; i < var_outputs.size(); ++i) {
               auto var = var_outputs[i];
               std::string name = output_names[i];
               imperative::VariableWrapper var_wrapper(name, var);
               auto shared_wrapper =
                   std::make_shared<imperative::VariableWrapper>(var_wrapper);
               auto shared_varbase =
                   std::make_shared<imperative::VarBase>(shared_wrapper);
               tensor_outputs.emplace_back(shared_varbase);
             }
             return tensor_outputs;
           })
      .def("info", &jit::ExecutorFunction::Info);

  py::class_<jit::FunctionInfo, std::shared_ptr<jit::FunctionInfo>>(
      *m, "FunctionInfo", R"DOC(FunctionInfo Class.)DOC")
      .def("name", &jit::FunctionInfo::FunctionName)
      .def("input_names", &jit::FunctionInfo::InputArgNames)
      .def("output_names", &jit::FunctionInfo::OutputArgNames);

  m->def("Load",
         [](const std::string &path, const platform::CPUPlace &cpu_place) {
           return paddle::jit::Load(path, cpu_place);
         });

  m->def("Load",
         [](const std::string &path, const platform::CUDAPlace &cuda_place) {
           return paddle::jit::Load(path, cuda_place);
         });
}

}  // namespace pybind
}  // namespace paddle
