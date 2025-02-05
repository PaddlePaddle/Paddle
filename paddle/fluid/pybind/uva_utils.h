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

#pragma once

#include <Python.h>
// Avoid a problem with copysign defined in pyconfig.h on Windows.
#ifdef copysign
#undef copysign
#endif

#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/tensor_utils.h"

namespace paddle {
namespace pybind {

static void tensor_uva(phi::DenseTensor *self_tensor, int device_id) {
  VLOG(4) << "Running in _uva interface.";
#if defined(PADDLE_WITH_CUDA)
  phi::DeviceContextPool &pool = phi::DeviceContextPool::Instance();
  auto *dev_ctx = pool.Get(phi::GPUPlace(device_id));
  VLOG(4) << "Init the DeviceContext, and the place is " << dev_ctx->GetPlace();
  // Register the cpu memory as the cuda host memory
  const auto &data_numel = self_tensor->numel();
  const size_t &need_allocate_size =
      data_numel * framework::SizeOfType(
                       framework::TransToProtoVarType(self_tensor->dtype()));
  void *data_ptr = self_tensor->data();
  auto result =
      cudaHostRegister(data_ptr, need_allocate_size, cudaHostRegisterDefault);
  if (cudaSuccess != result) {
    VLOG(4) << "UVA(unified virtual addressing) failed allocate:"
            << need_allocate_size << ", the error code:" << result;
  }
  // Get device pointer from the function of cudaHostGetDevicePointer
  void *cuda_device_pointer = nullptr;
  cudaHostGetDevicePointer(reinterpret_cast<void **>(&cuda_device_pointer),
                           reinterpret_cast<void *>(data_ptr),
                           0);

  // Reset the memory with device pointer
  std::shared_ptr<memory::allocation::Allocation> holder =
      std::make_shared<memory::allocation::Allocation>(
          cuda_device_pointer, need_allocate_size, phi::GPUPlace(device_id));
  self_tensor->ResetHolderWithType(holder, self_tensor->dtype());
#endif
}

}  // namespace pybind
}  // namespace paddle
