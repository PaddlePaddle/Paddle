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

#include "paddle/infrt/kernel/tensor_shape_kernels.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_os_ostream.h>

#include <iostream>

#include "paddle/infrt/host_context/kernel_registry.h"
#include "paddle/infrt/host_context/kernel_utils.h"
#include "paddle/infrt/tensor/tensor_shape.h"

namespace infrt {
namespace kernel {

void PrintShape(const tensor::TensorShape& shape) {
  llvm::raw_os_ostream oos(std::cout);
  oos << shape << '\n';
}

void RegisterTensorShapeKernels(host_context::KernelRegistry* registry) {
  registry->AddKernel("ts.print_shape", INFRT_KERNEL(PrintShape));
}

}  // namespace kernel
}  // namespace infrt
