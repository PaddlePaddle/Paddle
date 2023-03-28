// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/details/nan_inf_utils_detail.h"
#include "paddle/fluid/framework/details/nan_inf_utils.h"

#include <algorithm>
#include <unordered_map>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/phi/kernels/check_numerics_kernel.h"

namespace paddle {
namespace framework {
namespace details {

template <>
template <typename T>
void TensorCheckerVisitor<phi::GPUContext>::apply(
    typename std::enable_if<
        std::is_floating_point<T>::value ||
        std::is_same<T, ::paddle::platform::complex<float>>::value ||
        std::is_same<T, ::paddle::platform::complex<double>>::value>::type*)
    const {
  auto* dev_ctx = reinterpret_cast<phi::GPUContext*>(
      platform::DeviceContextPool::Instance().Get(tensor.place()));
  int dev_id = tensor.place().device;
  // Write log to file
  // auto file_path = GetNanPath();
  // if (file_path.size() > 0) {
  //   phi::DenseTensor cpu_tensor;
  //   platform::CPUPlace cpu_place;
  //   cpu_tensor.Resize(tensor.dims());
  //   // 1. copy from gpu to cpu
  //   paddle::framework::TensorCopySync(tensor, cpu_place, &cpu_tensor);
  //   const std::string debug_info =
  //       GetHintString<T>(op_type, var_name, place, dev_id);
  //   // 2. write log to file
  //   CheckNanInfCpuImpl(cpu_tensor.data<T>(), tensor.numel(), debug_info,
  //   "gpu"); return;
  // }

  phi::CheckNumericsKernel<T, phi::GPUContext>(
      *dev_ctx, tensor, op_type, var_name);
}

template <>
void tensor_check<phi::GPUContext>(const std::string& op_type,
                                   const std::string& var_name,
                                   const phi::DenseTensor& tensor,
                                   const platform::Place& place) {
  TensorCheckerVisitor<phi::GPUContext> vistor(
      op_type, var_name, tensor, place);
  VisitDataType(framework::TransToProtoVarType(tensor.dtype()), vistor);
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
