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
  std::string kernel_signature_map_str{"{"};
  for (const auto &op_kernel_pair : kernel_factory.kernels()) {
    std::string op_name = op_kernel_pair.first;
    const paddle::flat_hash_map<std::string, std::string> &kernel_name_map =
        phi::OpUtilsMap::Instance().base_kernel_name_map();
    for (auto &it : kernel_name_map) {
      if (it.second == op_name) {
        op_name = it.first;
        break;
      }
    }
    if (kernel_signature_map.Has(op_name)) {
      kernel_signature_map_str =
          kernel_signature_map_str + "\"" + op_kernel_pair.first + "\":{";
      auto &args = kernel_signature_map.Get(op_name).args;

      kernel_signature_map_str += "\"inputs\":[";
      auto inputs_ = std::get<0>(args);
      for (size_t i = 0; i < inputs_.size(); i++) {
        kernel_signature_map_str =
            kernel_signature_map_str + "\"" + inputs_[i] + "\",";
      }
      if (inputs_.size()) kernel_signature_map_str.pop_back();

      kernel_signature_map_str += "],\"attrs\":[";
      auto attrs_ = std::get<1>(args);
      for (size_t i = 0; i < attrs_.size(); i++) {
        kernel_signature_map_str =
            kernel_signature_map_str + "\"" + attrs_[i] + "\",";
      }
      if (attrs_.size()) kernel_signature_map_str.pop_back();
      kernel_signature_map_str += "],\"outputs\":[";
      auto outputs_ = std::get<2>(args);
      for (size_t i = 0; i < outputs_.size(); i++) {
        kernel_signature_map_str =
            kernel_signature_map_str + "\"" + outputs_[i] + "\",";
      }

      if (outputs_.size()) kernel_signature_map_str.pop_back();
      kernel_signature_map_str += "]},";
    }
  }
  kernel_signature_map_str.pop_back();
  kernel_signature_map_str += "}\n";
  std::cout << kernel_signature_map_str;
  return 0;
}
