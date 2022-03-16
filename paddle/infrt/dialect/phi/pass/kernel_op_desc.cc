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

#include "paddle/infrt/dialect/phi/pass/kernel_op_desc.h"
#include <glog/logging.h>
#include "paddle/infrt/dialect/phi/data_type.h"
#include "paddle/phi/kernels/declarations.h"

namespace infrt {

std::string getPhiTargetPrefix(TargetType target) {
  switch (target) {
    case TargetType::CPU:
      return "phi_cpu.";
    case TargetType::GPU:
      return "phi_gpu.";
    default:
      LOG(FATAL) << "UnSupported target type !";
      return std::string();
  }
}
std::string getPhiPrecisionSuffix(PrecisionType precision) {
  switch (precision) {
    case PrecisionType::FLOAT32:
      return ".float32";
    case PrecisionType::FLOAT16:
      return ".float16";
    case PrecisionType::FLOAT64:
      return ".float64";
    case PrecisionType::UINT8:
      return ".uint8";
    case PrecisionType::INT8:
      return ".int8";
    case PrecisionType::INT16:
      return ".int16";
    case PrecisionType::INT32:
      return ".int32";
    case PrecisionType::INT64:
      return ".int64";
    case PrecisionType::COMPLEX64:
      return ".complex64";
    case PrecisionType::COMPLEX128:
      return ".complex128";
    case PrecisionType::BOOL:
      return ".bool";
    default:
      LOG(FATAL) << "UnSupported precision type !";
      return std::string();
  }
}
std::string getPhiLayoutSuffix(LayoutType layout) {
  switch (layout) {
    case LayoutType::NCHW:
      return ".nchw";
    case LayoutType::NHWC:
      return ".nhwc";
    case LayoutType::ANY:
      return ".any";
    default:
      LOG(FATAL) << "UnSupported layout type !";
      return std::string();
  }
}

std::vector<PhiKernelDesc> GetCandidateKernels(
    std::string name, const std::vector<Place>& valid_palces) {
  std::vector<PhiKernelDesc> candidate_kernels;
  PhiKernelDesc phi_kernel_desc;
  phi::KernelKeyMap kernel_key_map =
      phi::KernelFactory::Instance().SelectKernelMap(name);
  for (Place place : valid_palces) {
    phi::KernelKey kernel_key = ConvertPlaceToPhi(place);
    if (kernel_key_map.find(kernel_key) == kernel_key_map.end()) {
      kernel_key = phi::KernelKey(kernel_key.backend(),
                                  phi::DataLayout::ALL_LAYOUT,
                                  kernel_key.dtype());
      if (kernel_key_map.find(kernel_key) == kernel_key_map.end()) continue;
      place.layout = LayoutType::ANY;
    }
    phi_kernel_desc.kernel_type = place;
    phi_kernel_desc.input_types.clear();
    phi_kernel_desc.output_types.clear();
    phi::KernelArgsDef args_def = kernel_key_map.at(kernel_key).args_def();
    const paddle::SmallVector<phi::TensorArgDef>& input_arg =
        args_def.input_defs();
    const paddle::SmallVector<phi::TensorArgDef>& output_arg =
        args_def.output_defs();
    for (auto tensor_arg : input_arg) {
      phi_kernel_desc.input_types.emplace_back(ConvertPlaceFromPhi(tensor_arg));
    }
    for (auto tensor_arg : output_arg) {
      phi_kernel_desc.output_types.emplace_back(
          ConvertPlaceFromPhi(tensor_arg));
    }
    candidate_kernels.emplace_back(phi_kernel_desc);
  }
  return candidate_kernels;
}

}  // namespace infrt
