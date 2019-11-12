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

#include "paddle/fluid/framework/details/var_utils.h"
#include "paddle/fluid/framework/details/var_utils_detail.h"

#include <algorithm>
#include "paddle/fluid/framework/selected_rows.h"

namespace paddle {
namespace framework {
namespace details {

// more detail see: 180 page of
// https://www.openmp.org/wp-content/uploads/OpenMP4.0.0.pdf
#pragma omp declare reduction(+ : paddle::platform::float16 : omp_out += omp_in)

template <typename T>
void CheckNanInf(const T* value, const size_t numel, int print_num,
                 const std::string& op_type, const std::string& var_name) {
  T sum = static_cast<T>(0.0);
#pragma omp parallel for simd reduction(+ : sum)
  for (size_t i = 0; i < numel; ++i) {
    sum += (value[i] - value[i]);
  }

  if (std::isnan(sum) || std::isinf(sum)) {
    // CPU print all value
    for (size_t i = 0; i < numel; ++i) {
      printf("idx:%lu value:%f\n", static_cast<uint64_t>(i),
             static_cast<float>(value[i]));
    }
    PADDLE_ENFORCE_EQ(1, 0,
                      "===ERROR: in [op=%s] [tensor=%s] find nan or inf===",
                      op_type, var_name);
  }
}

template <>
template <typename T>
void TensorCheckerVisitor<platform::CPUDeviceContext>::apply(
    typename std::enable_if<std::is_floating_point<T>::value>::type*) const {
  int print_num = 3;
  CheckNanInf(tensor_.data<T>(), tensor_.numel(), print_num, op_type_,
              var_name_);
}

template <>
void tensor_check<platform::CPUDeviceContext>(const std::string& op_type,
                                              const std::string& var_name,
                                              const framework::Tensor& tensor,
                                              const platform::Place& place) {
  TensorCheckerVisitor<platform::CPUDeviceContext> vistor(op_type, var_name,
                                                          tensor, place);
  VisitDataType(tensor.type(), vistor);
}

void EnforceNoNanOrInf(const std::string& op_type,
                       const framework::Scope& scope,
                       const std::string& var_name,
                       const platform::Place& place) {
  auto* var = scope.FindVar(var_name);
  PADDLE_ENFORCE_NOT_NULL(var, "can't find var:%s", var_name);

  const Tensor* tensor{nullptr};
  if (var->IsType<framework::LoDTensor>()) {
    tensor = &var->Get<framework::LoDTensor>();
  } else if (var->IsType<framework::SelectedRows>()) {
    tensor = &var->Get<framework::SelectedRows>().value();
  } else {
    VLOG(10) << var_name << " var_name need not to check";
    return;
  }

  if (tensor->memory_size() == 0) {
    VLOG(10) << var_name << " var_name need not to check, size == 0";
    return;
  }

  VLOG(10) << "begin check " << op_type << " var_name:" << var_name
           << ", place:" << tensor->place() << ", numel:" << tensor->numel();

  if (platform::is_gpu_place(tensor->place())) {
#ifdef PADDLE_WITH_CUDA
    tensor_check<platform::CUDADeviceContext>(op_type, var_name, *tensor,
                                              place);
#else
    PADDLE_THROW("PaddlePaddle should compile with GPU.");
#endif
    return;
  }

  tensor_check<platform::CPUDeviceContext>(op_type, var_name, *tensor, place);
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
