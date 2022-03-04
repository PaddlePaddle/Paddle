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
#include "paddle/phi/core/kernel_factory.h"
#include "paddle/phi/core/kernel_registry.h"
namespace infrt {

phi::Backend cvtTarget2Phi(TargetType target) {
  switch (target) {
    case TargetType::CPU:
      return phi::Backend::CPU;
    case TargetType::GPU:
      return phi::Backend::GPU;
    case TargetType::UNK:
      return phi::Backend::UNDEFINED;
  }
  return phi::Backend::UNDEFINED;
}

TargetType cvtTargetFromPhi(phi::Backend backend) {
  switch (backend) {
    case phi::Backend::CPU:
      return TargetType::CPU;
    case phi::Backend::GPU:
      return TargetType::GPU;
    default:
      return TargetType::UNK;
  }
}

phi::DataType cvtPrecision2Phi(PrecisionType precision) {
#define CONVERT_PRECISION_TO_PHI(Precision) \
  case PrecisionType::Precision:            \
    return phi::DataType::Precision;

  switch (precision) {
    CONVERT_PRECISION_TO_PHI(FLOAT32)
    CONVERT_PRECISION_TO_PHI(FLOAT16)
    CONVERT_PRECISION_TO_PHI(FLOAT64)
    CONVERT_PRECISION_TO_PHI(UINT8)
    CONVERT_PRECISION_TO_PHI(INT8)
    CONVERT_PRECISION_TO_PHI(INT16)
    CONVERT_PRECISION_TO_PHI(INT32)
    CONVERT_PRECISION_TO_PHI(INT64)
    CONVERT_PRECISION_TO_PHI(COMPLEX64)
    CONVERT_PRECISION_TO_PHI(COMPLEX128)
    CONVERT_PRECISION_TO_PHI(BOOL)
    default:
      return phi::DataType::UNDEFINED;
  }
#undef CONVERT_PRECISION_TO_PHI
}

PrecisionType cvtPrecisionFromPhi(phi::DataType datatype) {
#define CONVERT_PRECISION_FROM_PHI(Precision) \
  case phi::DataType::Precision:              \
    return PrecisionType::Precision;

  switch (datatype) {
    CONVERT_PRECISION_FROM_PHI(FLOAT32)
    CONVERT_PRECISION_FROM_PHI(FLOAT16)
    CONVERT_PRECISION_FROM_PHI(FLOAT64)
    CONVERT_PRECISION_FROM_PHI(UINT8)
    CONVERT_PRECISION_FROM_PHI(INT8)
    CONVERT_PRECISION_FROM_PHI(INT16)
    CONVERT_PRECISION_FROM_PHI(INT32)
    CONVERT_PRECISION_FROM_PHI(INT64)
    CONVERT_PRECISION_FROM_PHI(COMPLEX64)
    CONVERT_PRECISION_FROM_PHI(COMPLEX128)
    CONVERT_PRECISION_FROM_PHI(BOOL)
    default:
      return PrecisionType::UNK;
  }
#undef CONVERT_PRECISION_FROM_PHI
}

phi::DataLayout cvtLayout2Phi(LayoutType layout) {
  switch (layout) {
    case LayoutType::NCHW:
      return phi::DataLayout::NCHW;
    case LayoutType::NHWC:
      return phi::DataLayout::NHWC;
    case LayoutType::ANY:
      return phi::DataLayout::ANY;
    default:
      return phi::DataLayout::UNDEFINED;
  }
}

LayoutType cvtLayoutFromPhi(phi::DataLayout layout) {
  switch (layout) {
    case phi::DataLayout::NCHW:
      return LayoutType::NCHW;
    case phi::DataLayout::NHWC:
      return LayoutType::NHWC;
    case phi::DataLayout::ANY:
      return LayoutType::ANY;
    default:
      return LayoutType::UNK;
  }
}

phi::KernelKey ConvertPlace2Phi(const Place& place) {
  return phi::KernelKey(cvtTarget2Phi(place.target),
                        cvtLayout2Phi(place.layout),
                        cvtPrecision2Phi(place.precision));
}

Place ConvertPlaceFromPhi(phi::TensorArgDef tensor_arg) {
  return Place(cvtTargetFromPhi(tensor_arg.backend),
               cvtPrecisionFromPhi(tensor_arg.dtype),
               cvtLayoutFromPhi(tensor_arg.layout));
}

std::vector<PhiKernelDesc> GetCandidateKernels(
    std::string name, const std::vector<Place>& valid_palces) {
  std::vector<PhiKernelDesc> candidate_kernels;
  PhiKernelDesc phi_kernel_desc;
  phi::KernelKeyMap kernel_key_map =
      phi::KernelFactory::Instance().SelectKernelMap(name);
  for (const Place& place : valid_palces) {
    phi::KernelKey kernel_key = ConvertPlace2Phi(place);
    if (kernel_key_map.find(kernel_key) == kernel_key_map.end()) {
      kernel_key = phi::KernelKey(kernel_key.backend(),
                                  phi::DataLayout::ALL_LAYOUT,
                                  kernel_key.dtype());
      if (kernel_key_map.find(kernel_key) == kernel_key_map.end()) continue;
    }
    phi_kernel_desc.kernel_type = place;
    phi_kernel_desc.inputs_types.clear();
    phi_kernel_desc.output_types.clear();
    phi::KernelArgsDef args_def = kernel_key_map.at(kernel_key).args_def();
    const paddle::SmallVector<phi::TensorArgDef>& input_arg =
        args_def.input_defs();
    const paddle::SmallVector<phi::TensorArgDef>& output_arg =
        args_def.output_defs();
    for (auto tensor_arg : input_arg) {
      phi_kernel_desc.inputs_types.emplace_back(
          ConvertPlaceFromPhi(tensor_arg));
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
