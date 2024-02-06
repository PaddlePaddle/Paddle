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

#include "glog/logging.h"
#include "paddle/common/flags.h"
#include "paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_functions.h"
#include "paddle/fluid/eager/api/manual/eager_manual/nodes/nodes.h"
#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/eager/nan_inf_utils.h"
#include "paddle/fluid/eager/utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/phi/api/all.h"
#include "paddle/phi/api/backward/sparse_bw_api.h"
#include "paddle/phi/api/lib/api_custom_impl.h"

COMMON_DECLARE_bool(check_nan_inf);

paddle::small_vector<std::vector<paddle::Tensor>, egr::kSlotSmallVectorSize>
CastGradNode::operator()(paddle::small_vector<std::vector<paddle::Tensor>,
                                              egr::kSlotSmallVectorSize>& grads,
                         bool create_graph,
                         bool is_new_grad) {
  VLOG(3) << "Running AD API GRAD: "
          << "cast_grad";

  // Fill Zero For GradIn Tensors

  // Apply Gradient Hooks
  auto hooked_grads = ApplyGradientHooks(grads);

  // Collect GradIn Tensors, Attrs and Recovered TensorWrappers
  auto x = egr::EagerUtils::RecoverTensorWrapper(&this->x_);
  auto& out_grad = hooked_grads[0][0];
  // Prepare Grad function call

  const auto& out_metas = OutputMeta();
  paddle::small_vector<std::vector<paddle::Tensor>, egr::kSlotSmallVectorSize>
      returns(1);
  for (int i = 0; i < 1; ++i) {
    out_metas[i].size() == 0 ? returns[i].resize(1)
                             : returns[i].resize(out_metas[i].size());
  }

  auto* api_output_0 =
      (out_metas[0].empty() || out_metas[0][0].IsStopGradient())
          ? nullptr
          : &returns[0][0];
  // Runtime check if we need next grad
  bool trace_backward = egr::Controller::Instance().HasGrad() && create_graph;

  // Set DistAttr of Out Tensor for semi-auto parallel

  // Inplace Check

  // Inplace Strategy

  VLOG(5) << "Running C++ API: "
          << "cast_grad";
  // Before log info

  if (VLOG_IS_ON(3)) {
    const char* INPUT_PRINT_TEMPLATE = "{ Input: [%s]} ";

    std::string input_str = "";
    std::string output_str = "";
    const char* TENSOR_OUT_GRAD_TEMPLATE = " \n( out_grad , [%s]), ";
    std::string input_out_grad_str = paddle::string::Sprintf(
        TENSOR_OUT_GRAD_TEMPLATE, egr::EagerUtils::TensorStr(out_grad));
    input_str += input_out_grad_str;
    const char* TENSOR_X_TEMPLATE = " \n( x , [%s]), ";
    std::string input_x_str = paddle::string::Sprintf(
        TENSOR_X_TEMPLATE, egr::EagerUtils::TensorStr(x));
    input_str += input_x_str;
    VLOG(3) << paddle::string::Sprintf(INPUT_PRINT_TEMPLATE, input_str);
  }

  // Call grad_api function

  if (trace_backward) {
    auto api_output = cast_ad_func(out_grad, x.dtype());
    *api_output_0 = api_output;
  } else {
    auto api_output = paddle::experimental::cast(out_grad, x.dtype());
    *api_output_0 = api_output;
  }

  // Check NaN and Inf id needed

  if (FLAGS_check_nan_inf) {
    try {
      egr::CheckTensorHasNanOrInf("cast_grad", returns);
    } catch (...) {
      LOG(WARNING) << "There are nan/inf in (cast_grad)";
      auto forward_trace = GetForwardTrace();
      std::cout << forward_trace << std::endl;
      std::rethrow_exception(std::current_exception());
    }
  }

  // Get GradOut autograd_meta

  auto& x_grad = returns[0][0];
  egr::AutogradMeta* x_grad_autograd_meta =
      returns[0][0].initialized() ? egr::EagerUtils::autograd_meta(&x_grad)
                                  : nullptr;
  if (x_grad_autograd_meta) x_grad_autograd_meta->SetStopGradient(false);

  // Create Grad Node

  VLOG(4) << "Finish AD API GRAD: cast_grad";
  VLOG(6) << "gradnode_ptr = " << this;
  // LOG IF DEBUG

  if (VLOG_IS_ON(4)) {
    const char* INPUT_PRINT_TEMPLATE = "{ Input: [%s],  \n Output: [%s] } ";

    std::string input_str = "";
    std::string output_str = "";
    const char* TENSOR_OUT_GRAD_TEMPLATE = " \n( out_grad , [%s]), ";
    std::string input_out_grad_str = paddle::string::Sprintf(
        TENSOR_OUT_GRAD_TEMPLATE, egr::EagerUtils::TensorStr(out_grad));
    input_str += input_out_grad_str;
    const char* TENSOR_X_TEMPLATE = " \n( x , [%s]), ";
    std::string input_x_str = paddle::string::Sprintf(
        TENSOR_X_TEMPLATE, egr::EagerUtils::TensorStr(x));
    input_str += input_x_str;
    const char* TENSOR_X_GRAD_TEMPLATE = " \n ( x_grad , [%s]), ";
    std::string output_x_grad_str = paddle::string::Sprintf(
        TENSOR_X_GRAD_TEMPLATE, egr::EagerUtils::TensorStr(x_grad));
    output_str += output_x_grad_str;
    VLOG(4) << paddle::string::Sprintf(
        INPUT_PRINT_TEMPLATE, input_str, output_str);
  }

  // Return
  if (NeedComplexToRealConversion()) HandleComplexGradToRealGrad(&returns);
  return returns;
}

namespace sparse {
paddle::small_vector<std::vector<paddle::Tensor>, egr::kSlotSmallVectorSize>
CastGradNode::operator()(paddle::small_vector<std::vector<paddle::Tensor>,
                                              egr::kSlotSmallVectorSize>& grads,
                         bool create_graph,
                         bool is_new_grad) {
  VLOG(3) << "Running AD API GRAD: "
          << "cast_grad";

  // Fill Zero For GradIn Tensors

  // Apply Gradient Hooks
  auto hooked_grads = ApplyGradientHooks(grads);

  // Collect GradIn Tensors, Attrs and Recovered TensorWrappers
  auto x = egr::EagerUtils::RecoverTensorWrapper(&this->x_);
  auto& out_grad = hooked_grads[0][0];
  auto& value_dtype = this->value_dtype_;
  // Prepare Grad function call

  const auto& out_metas = OutputMeta();
  paddle::small_vector<std::vector<paddle::Tensor>, egr::kSlotSmallVectorSize>
      returns(1);
  for (int i = 0; i < 1; ++i) {
    out_metas[i].size() == 0 ? returns[i].resize(1)
                             : returns[i].resize(out_metas[i].size());
  }

  auto* api_output_0 =
      (out_metas[0].empty() || out_metas[0][0].IsStopGradient())
          ? nullptr
          : &returns[0][0];
  // Runtime check if we need next grad
  bool trace_backward = egr::Controller::Instance().HasGrad() && create_graph;

  // Set DistAttr of Out Tensor for semi-auto parallel

  if (IsRunAutoParallel()) {
    egr::EagerUtils::SetGradOutputDistAttr(out_metas, {0}, api_output_0);
  }

  // Inplace Check

  // Inplace Strategy

  VLOG(5) << "Running C++ API: "
          << "cast_grad";
  // Before log info

  if (VLOG_IS_ON(3)) {
    const char* INPUT_PRINT_TEMPLATE = "{ Input: [%s]} ";

    std::string input_str = "";
    std::string output_str = "";
    const char* TENSOR_OUT_GRAD_TEMPLATE = " \n( out_grad , [%s]), ";
    std::string input_out_grad_str = paddle::string::Sprintf(
        TENSOR_OUT_GRAD_TEMPLATE, egr::EagerUtils::TensorStr(out_grad));
    input_str += input_out_grad_str;
    const char* TENSOR_X_TEMPLATE = " \n( x , [%s]), ";
    std::string input_x_str = paddle::string::Sprintf(
        TENSOR_X_TEMPLATE, egr::EagerUtils::TensorStr(x));
    input_str += input_x_str;
    VLOG(3) << paddle::string::Sprintf(INPUT_PRINT_TEMPLATE, input_str);
  }

  // Call grad_api function

  paddle::experimental::sparse::cast_grad(
      x, out_grad, value_dtype, api_output_0);
  // Check NaN and Inf id needed

  if (FLAGS_check_nan_inf) {
    try {
      egr::CheckTensorHasNanOrInf("cast_grad", returns);
    } catch (...) {
      LOG(WARNING) << "There are nan/inf in (cast_grad)";
      auto forward_trace = GetForwardTrace();
      std::cout << forward_trace << std::endl;
      std::rethrow_exception(std::current_exception());
    }
  }

  // Get GradOut autograd_meta

  auto& x_grad = returns[0][0];
  egr::AutogradMeta* x_grad_autograd_meta =
      returns[0][0].initialized() ? egr::EagerUtils::autograd_meta(&x_grad)
                                  : nullptr;
  if (x_grad_autograd_meta) x_grad_autograd_meta->SetStopGradient(false);

  // Create Grad Node
  if (trace_backward) {
    PADDLE_THROW(phi::errors::Unavailable(
        "The Op cast_grad doesn't have any grad"
        "op. If you don't intend calculating higher order"
        "derivatives, please set `create_graph`to False."));
  }
  VLOG(4) << "Finish AD API GRAD: cast_grad";
  VLOG(6) << "gradnode_ptr = " << this;
  // LOG IF DEBUG

  if (VLOG_IS_ON(4)) {
    const char* INPUT_PRINT_TEMPLATE = "{ Input: [%s],  \n Output: [%s] } ";

    std::string input_str = "";
    std::string output_str = "";
    const char* TENSOR_OUT_GRAD_TEMPLATE = " \n( out_grad , [%s]), ";
    std::string input_out_grad_str = paddle::string::Sprintf(
        TENSOR_OUT_GRAD_TEMPLATE, egr::EagerUtils::TensorStr(out_grad));
    input_str += input_out_grad_str;
    const char* TENSOR_X_TEMPLATE = " \n( x , [%s]), ";
    std::string input_x_str = paddle::string::Sprintf(
        TENSOR_X_TEMPLATE, egr::EagerUtils::TensorStr(x));
    input_str += input_x_str;
    const char* TENSOR_X_GRAD_TEMPLATE = " \n ( x_grad , [%s]), ";
    std::string output_x_grad_str = paddle::string::Sprintf(
        TENSOR_X_GRAD_TEMPLATE, egr::EagerUtils::TensorStr(x_grad));
    output_str += output_x_grad_str;
    VLOG(4) << paddle::string::Sprintf(
        INPUT_PRINT_TEMPLATE, input_str, output_str);
  }

  // Return
  if (NeedComplexToRealConversion()) HandleComplexGradToRealGrad(&returns);
  return returns;
}
}  // namespace sparse
