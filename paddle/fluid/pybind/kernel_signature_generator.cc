// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include <iostream>
#include <string>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/phi_utils.h"
#include "paddle/fluid/pybind/pybind.h"  // NOLINT
#include "paddle/phi/core/compat/op_utils.h"
#include "paddle/phi/core/kernel_factory.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/declarations.h"

// print names of kernel function params with json format:
// {
// "norm":{
//   "inputs":[
//     "X"
//   ],
//   "attrs":[
//     "axis",
//     "epsilon",
//     "is_test"
//   ],
//   "outputs":[
//     "Norm",
//     "Out"
//   ]
// },
// ...
// }
int main(int argc, char **argv) {
  paddle::framework::InitDefaultKernelSignatureMap();
  auto &kernel_signature_map = phi::DefaultKernelSignatureMap::Instance();
  auto &kernel_factory = phi::KernelFactory::Instance();
  std::cout << "{";
  for (const auto &op_kernel_pair : kernel_factory.kernels()) {
    if (kernel_signature_map.Has(op_kernel_pair.first)) {
      std::cout << "\"" << op_kernel_pair.first << "\":{";
      auto &args = kernel_signature_map.Get(op_kernel_pair.first).args;

      std::cout << "\"inputs\":[";
      auto inputs_ = std::get<0>(args);
      if (inputs_.size() > 0) std::cout << inputs_[0];
      for (size_t i = 1; i < inputs_.size(); i++) {
        std::cout << ",\"" << inputs_[i] << "\"";
      }

      std::cout << "],\"attrs\":[";
      auto attrs_ = std::get<1>(args);
      if (attrs_.size() > 0) std::cout << attrs_[0];
      for (size_t i = 1; i < attrs_.size(); i++) {
        std::cout << ",\"" << attrs_[i] << "\"";
      }

      std::cout << "],\"outputs\":[";
      auto outputs_ = std::get<2>(args);
      for (size_t i = 1; i < outputs_.size(); i++) {
        std::cout << ",\"" << outputs_[i] << "\"";
      }

      std::cout << "]},";
    }
  }
  std::cout << "}" << std::endl;
  return 0;
}
