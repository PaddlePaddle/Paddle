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
#include "paddle/fluid/framework/pten_utils.h"
#include "paddle/fluid/pybind/pybind.h"  // NOLINT
#include "paddle/pten/core/compat/op_utils.h"
#include "paddle/pten/core/kernel_factory.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/kernels/declarations.h"

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
  auto &kernel_signature_map = pten::DefaultKernelSignatureMap::Instance();
  auto &kernel_factory = pten::KernelFactory::Instance();
  std::cout << "{";
  for (const auto &op_kernel_pair : kernel_factory.kernels()) {
    if (kernel_signature_map.Has(op_kernel_pair.first)) {
      std::cout << "\"" << op_kernel_pair.first << "\":{";
      auto &args = kernel_signature_map.Get(op_kernel_pair.first).args;
      std::cout << "\"inputs\":[";
      for (auto name : std::get<0>(args)) {
        std::cout << "\"" << name << "\",";
      }
      if (std::get<0>(args).size() > 0) std::cout << "\b";
      std::cout << "],\"attrs\":[";
      for (auto name : std::get<1>(args)) {
        std::cout << "\"" << name << "\",";
      }
      if (std::get<1>(args).size() > 0) std::cout << "\b";
      std::cout << "],\"outputs\":[";
      for (auto name : std::get<2>(args)) {
        std::cout << "\"" << name << "\",";
      }
      if (std::get<2>(args).size() > 0) std::cout << "\b";
      std::cout << "]},";
    }
  }
  std::cout << "\b}" << std::endl;
  return 0;
}
