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
#include <unordered_map>
#include <unordered_set>
#include "paddle/fluid/framework/selected_rows.h"

namespace paddle {
namespace framework {
namespace details {

const std::unordered_set<std::string> op_nan_inf_white_list = {
    "coalesce_tensor", /* This Op will alloc tensor, and may not init space */
};

const std::unordered_map<std::string, std::vector<std::string>>
    op_var_nan_inf_white_list = {
        /* encoded & gather var consist of idx&val, can't judge directly */
        {"dgc", {"__dgc_encoded__", "__dgc_gather__"}},
};

template <typename T>
static void PrintNanInf(const T* value, const size_t numel, int print_num,
                        const std::string& op_type,
                        const std::string& var_name) {
  // CPU print all value
  for (size_t i = 0; i < numel; ++i) {
    printf("index:%lu value:%f\n", static_cast<uint64_t>(i),
           static_cast<float>(value[i]));
  }
  bool has_nan_inf = true;
  PADDLE_ENFORCE_EQ(has_nan_inf, false,
                    "===ERROR: in [op=%s] [tensor=%s] find nan or inf===",
                    op_type, var_name);
}

// openmp 4.0, reduction with fp16
#if defined _OPENMP && _OPENMP >= 201307
// more detail see: 180 page of
// https://www.openmp.org/wp-content/uploads/OpenMP4.0.0.pdf
#pragma omp declare reduction(+ : paddle::platform::float16 : omp_out += omp_in)
#endif

template <typename T>
static void CheckNanInf(const T* value, const size_t numel, int print_num,
                        const std::string& op_type,
                        const std::string& var_name) {
  T sum = static_cast<T>(0.0);
#if defined _OPENMP && _OPENMP >= 201307
#pragma omp parallel for simd reduction(+ : sum)
#elif defined _OPENMP
#pragma omp parallel for reduction(+ : sum)
#endif
  for (size_t i = 0; i < numel; ++i) {
    sum += (value[i] - value[i]);
  }

  if (std::isnan(sum) || std::isinf(sum)) {
    PrintNanInf(value, numel, print_num, op_type, var_name);
  }
}

#if defined _OPENMP && _OPENMP >= 201307
// openmp4.0 not need to specialization fp16
#elif defined _OPENMP
template <>
static void CheckNanInf<paddle::platform::float16>(
    const paddle::platform::float16* value, const size_t numel, int print_num,
    const std::string& op_type, const std::string& var_name) {
  float sum = 0.0f;
#pragma omp parallel for reduction(+ : sum)
  for (size_t i = 0; i < numel; ++i) {
    sum += static_cast<float>(value[i] - value[i]);
  }

  if (std::isnan(sum) || std::isinf(sum)) {
    PrintNanInf(value, numel, print_num, op_type, var_name);
  }
}
#endif

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

void CheckVarHasNanOrInf(const std::string& op_type,
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

void CheckOpHasNanOrInf(const framework::OperatorBase& op,
                        const framework::Scope& exec_scope,
                        const platform::Place& place) {
  if (op_nan_inf_white_list.count(op.Type()) != 0) return;

  if (op_var_nan_inf_white_list.count(op.Type()) == 0) {
    // NOTE. vname may destruct in the end of this func.
    for (auto& vname : op.OutputVars(true)) {
      auto* var = exec_scope.FindVar(vname);
      if (var == nullptr) continue;
      CheckVarHasNanOrInf(op.Type(), exec_scope, vname, place);
    }
  } else {
    for (auto& vname : op.OutputVars(true)) {
      bool need_check = true;
      for (auto& white_vname : op_var_nan_inf_white_list.at(op.Type())) {
        if (vname.find(white_vname) != std::string::npos) {
          need_check = false;
          break;
        }
      }
      if (!need_check) continue;
      auto* var = exec_scope.FindVar(vname);
      if (var == nullptr) continue;
      CheckVarHasNanOrInf(op.Type(), exec_scope, vname, place);
    }
  }
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
