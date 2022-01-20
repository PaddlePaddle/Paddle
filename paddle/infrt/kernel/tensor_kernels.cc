// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/infrt/kernel/tensor_kernels.h"

#include <iostream>
#include <vector>

#include "paddle/infrt/common/global.h"
#include "paddle/infrt/host_context/kernel_registry.h"
#include "paddle/infrt/host_context/kernel_utils.h"
#include "paddle/infrt/tensor/dense_host_tensor.h"
#include "paddle/infrt/tensor/dense_tensor_view.h"
#include "paddle/infrt/tensor/tensor_map.h"
#include "paddle/infrt/tensor/tensor_shape.h"

namespace infrt {
namespace kernel {
using namespace host_context;  // NOLINT
using namespace tensor;        // NOLINT

/// ===== Kernel begin ====

template <typename T>
DenseHostTensor CreateUninitTensor(Attribute<std::vector<int64_t>> shape) {
  const auto &shape_data = shape.get();
  auto array = llvm::ArrayRef<int64_t>(shape_data.data(), shape_data.size());
  auto type = GetDType<T>();
  return DenseHostTensor(TensorShape(array), type);
}

void PrintTensor(const DenseHostTensor &tensor) {
  std::cout << tensor << std::endl;
}

template <typename T>
void FillTensorWithConstant(DenseHostTensor *tensor, Attribute<T> v) {
  MutableDTArrayView<T>(tensor).Fill(v.get());
}

TensorMap LoadParams(const std::string &path) {
  return *(infrt::tensor::LoadParams(path));
}

DenseHostTensor GetParam(TensorMap map, Attribute<std::string> nameAttr) {
  auto &name = nameAttr.get();
  return *(map[name]);
}

DenseHostTensor ShallowCopyTensor(DenseHostTensor v) { return v; }

/// ===== Kernel end ====

void RegisterTensorKernels(host_context::KernelRegistry *registry) {
  registry->AddKernel("dt.create_uninit_tensor.f32",
                      INFRT_KERNEL(CreateUninitTensor<float>));
  registry->AddKernelAttrNameList("dt.create_uninit_tensor.f32", {"shape"});
  registry->AddKernel("dt.print_tensor", INFRT_KERNEL(PrintTensor));
  registry->AddKernel("dt.fill_tensor_with_constant.f32",
                      INFRT_KERNEL(FillTensorWithConstant<float>));
  registry->AddKernel("dt.fill_tensor_with_constant.f64",
                      INFRT_KERNEL(FillTensorWithConstant<double>));
  registry->AddKernel("dt.load_params", INFRT_KERNEL(LoadParams));
  registry->AddKernel("dt.get_param", INFRT_KERNEL(GetParam));
  registry->AddKernel("dt.shallow_copy_tensor",
                      INFRT_KERNEL(ShallowCopyTensor));
}

}  // namespace kernel
}  // namespace infrt
