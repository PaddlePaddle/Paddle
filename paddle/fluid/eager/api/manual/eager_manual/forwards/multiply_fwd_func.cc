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

#include "paddle/common/flags.h"
#include "paddle/fluid/eager/api/manual/eager_manual/dygraph_forward_api.h"
#include "paddle/fluid/eager/api/manual/eager_manual/nodes/nodes.h"
#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/eager/eager_layout_auto_tune.h"
#include "paddle/fluid/eager/nan_inf_utils.h"
#include "paddle/fluid/eager/type_promotion_utils.h"
#include "paddle/fluid/imperative/amp_utils.h"
#include "paddle/phi/api/include/sparse_api.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/common/type_promotion.h"
#include "paddle/phi/core/platform/profiler/event_tracing.h"

COMMON_DECLARE_bool(check_nan_inf);

bool check_if_support_elementwise_mul_mem_opt(const std::string& device_type) {
  // TODO(@gexiao): replace this function with api implemented at custom repo
  if (device_type == "npu") {
    return true;
  } else {
    return false;
  }
}

paddle::Tensor multiply_ad_func(const paddle::Tensor& x,
                                const paddle::Tensor& y) {
  FLAGS_tensor_operants_mode = "eager";
  VLOG(3) << "Running AD API: "
          << "multiply";
  // Dygraph Record Event
  phi::RecordEvent dygraph_entrance_record_event(
      "multiply dygraph", phi::TracerEventType::Operator, 1);

  // AMP Logic
  if (egr::Controller::Instance().GetAMPLevel() !=
      paddle::imperative::AmpLevel::O0) {
    VLOG(5) << "Check and Prepare For AMP";
    auto op_name = phi::TransToFluidOpName("multiply");
    paddle::small_vector<std::vector<paddle::Tensor>, egr::kSlotSmallVectorSize>
        amp_tensors_vector = {{x}, {y}};

    auto amp_dst_dtype =
        paddle::imperative::GetAmpDestDtype(op_name, amp_tensors_vector);

    auto new_x =
        paddle::imperative::AmpAutoCast("x", x, amp_dst_dtype, op_name);
    auto new_y =
        paddle::imperative::AmpAutoCast("y", y, amp_dst_dtype, op_name);

    {
      paddle::imperative::AutoCastGuard guard(
          egr::Controller::Instance().GetCurrentAmpAttrs(),
          paddle::imperative::AmpLevel::O0);
      return multiply_ad_func(new_x, new_y);
    }
  }

  // Type promotion Logic
  if (phi::NeedTypePromotion(
          "multiply", x.dtype(), y.dtype(), x.shape(), y.shape())) {
    VLOG(5) << "got different data type, run type promotion automatically.";
    LOG_FIRST_N(WARNING, 1)
        << "got different data type, run type promotion "
           "automatically, this may cause data type been changed.";
    auto op_name = phi::TransToFluidOpName("multiply");
    auto promotion_type = phi::GetPromoteDtype(
        op_name, x.dtype(), y.dtype(), x.shape(), y.shape());

    auto new_x = egr::PromoteCast("x", x, promotion_type);
    auto new_y = egr::PromoteCast("y", y, promotion_type);

    return multiply_ad_func(new_x, new_y);
  }

  // Layout autotune

  if (egr::Controller::Instance().UseLayoutAutoTune()) {
    paddle::small_vector<std::vector<paddle::Tensor>, egr::kSlotSmallVectorSize>
        tensors_vector = {{x}, {y}};

    auto op_name = phi::TransToFluidOpName("multiply");
    auto transformer = egr::EagerLayoutAutotune(op_name, tensors_vector);
    auto new_x = transformer->TransInTensor("x", x);
    auto new_y = transformer->TransInTensor("y", y);

    VLOG(5) << "Check and Prepare For LAYOUT " << op_name;
    paddle::imperative::LayoutAutotuneGuard guard(
        egr::Controller::Instance().GetCurrentTracer(), false);
    paddle::Tensor out = multiply_ad_func(new_x, new_y);

    transformer->SetOutTensorLayout(&out);

    // Returns
    return out;
  }

  // Get Input AutoGradMeta
  egr::AutogradMeta* x_autograd_meta =
      egr::EagerUtils::nullable_autograd_meta(x);
  egr::AutogradMeta* y_autograd_meta =
      egr::EagerUtils::nullable_autograd_meta(y);

  VLOG(5) << "Running C++ API: "
          << "multiply";
  // Before log info

  if (VLOG_IS_ON(3)) {
    const char* INPUT_PRINT_TEMPLATE = "{ Input: [%s]} ";

    std::string input_str = "";
    std::string output_str = "";
    const char* TENSOR_X_TEMPLATE = " \n( x , [%s]), ";
    std::string input_x_str = paddle::string::Sprintf(
        TENSOR_X_TEMPLATE, egr::EagerUtils::TensorStr(x));
    input_str += input_x_str;
    const char* TENSOR_Y_TEMPLATE = " \n( y , [%s]), ";
    std::string input_y_str = paddle::string::Sprintf(
        TENSOR_Y_TEMPLATE, egr::EagerUtils::TensorStr(y));
    input_str += input_y_str;
    VLOG(3) << paddle::string::Sprintf(INPUT_PRINT_TEMPLATE, input_str);
  }

  // Forward API Call
  auto api_result = paddle::experimental::multiply(x, y);
  // Check NaN and Inf if needed

  if (FLAGS_check_nan_inf) {
    egr::CheckTensorHasNanOrInf("multiply", api_result);
  }

  // Get Outputs
  auto& out = api_result;

  // Get Output AutoGradMeta
  egr::AutogradMeta* out_autograd_meta = egr::EagerUtils::autograd_meta(&out);
  bool trace_backward = egr::Controller::Instance().HasGrad();
  bool require_any_grad = egr::EagerUtils::ComputeRequireGrad(
      trace_backward, x_autograd_meta, y_autograd_meta);

  // Check Inplace if needed

  // Node Creation
  if (require_any_grad) {
    phi::RecordEvent node_creation_record_event(
        "multiply node_creation", phi::TracerEventType::OperatorInner, 1);

    egr::EagerUtils::PassStopGradient(false, out_autograd_meta);

    // Node Construction
    auto grad_node = std::shared_ptr<MultiplyGradNode>(  // NOLINT
        new MultiplyGradNode(1, 2));
    // Set for forward trace
    if (FLAGS_check_nan_inf) {
      grad_node->SetForwardTrace(egr::Controller::Instance().GetPythonStack());
    }
    // SetAttributes if needed
    grad_node->SetAttribute_axis(-1);
#ifdef PADDLE_WITH_CUSTOM_DEVICE
    if (check_if_support_elementwise_mul_mem_opt(x.place().GetDeviceType())) {
#else
    if (phi::is_gpu_place(x.place())) {
#endif
      if (x_autograd_meta != nullptr && x_autograd_meta->StopGradient() &&
          y_autograd_meta != nullptr && !y_autograd_meta->StopGradient()) {
        grad_node->SetTensorWrapper_x(x);
        grad_node->SetTensorWrapperNoNeedBuffer_y(y);
      } else if (x_autograd_meta != nullptr &&
                 !x_autograd_meta->StopGradient() &&
                 y_autograd_meta != nullptr &&
                 y_autograd_meta->StopGradient()) {
        grad_node->SetTensorWrapperNoNeedBuffer_x(x);
        grad_node->SetTensorWrapper_y(y);
      } else {
        grad_node->SetTensorWrapper_x(x);
        grad_node->SetTensorWrapper_y(y);
      }
    } else {
      grad_node->SetTensorWrapper_x(x);
      grad_node->SetTensorWrapper_y(y);
    }
    // SetGradOutMeta & SetEdges
    grad_node->SetGradOutMeta(x, 0);
    grad_node->SetGradOutMeta(y, 1);
    // SetOutRank & SetHistory & SetGradInMeta
    if (out_autograd_meta) {
      egr::EagerUtils::SetOutRankWithSlot(out_autograd_meta, 0);
    }
    if (out_autograd_meta) {
      egr::EagerUtils::SetHistory(out_autograd_meta, grad_node);
    }
    grad_node->SetGradInMeta(out, 0);
    // Set TensorWrappers for Forward Outputs if needed
  }

  VLOG(4) << "Finish AD API: multiply";
  // LOG IF DEBUG

  if (VLOG_IS_ON(4)) {
    const char* INPUT_PRINT_TEMPLATE = "{ Input: [%s],  \n Output: [%s] } ";

    std::string input_str = "";
    std::string output_str = "";
    const char* TENSOR_X_TEMPLATE = " \n( x , [%s]), ";
    std::string input_x_str = paddle::string::Sprintf(
        TENSOR_X_TEMPLATE, egr::EagerUtils::TensorStr(x));
    input_str += input_x_str;
    const char* TENSOR_Y_TEMPLATE = " \n( y , [%s]), ";
    std::string input_y_str = paddle::string::Sprintf(
        TENSOR_Y_TEMPLATE, egr::EagerUtils::TensorStr(y));
    input_str += input_y_str;
    const char* TENSOR_OUT_TEMPLATE = " \n( out , [%s]), ";
    std::string output_out_str = paddle::string::Sprintf(
        TENSOR_OUT_TEMPLATE, egr::EagerUtils::TensorStr(out));
    output_str += output_out_str;
    VLOG(4) << paddle::string::Sprintf(
        INPUT_PRINT_TEMPLATE, input_str, output_str);
  }

  // Returns
  return out;
}

paddle::Tensor& multiply__ad_func(paddle::Tensor& x,  // NOLINT
                                  const paddle::Tensor& y) {
  FLAGS_tensor_operants_mode = "eager";
  VLOG(3) << "Running AD API: "
          << "multiply_";
  // Dygraph Record Event
  phi::RecordEvent dygraph_entrance_record_event(
      "multiply_ dygraph", phi::TracerEventType::Operator, 1);

  // AMP Logic

  VLOG(5)
      << " No AMP for multiply__ad_func because it is a inplace or cast api. ";

  // Type promotion Logic
  if (phi::NeedTypePromotion(
          "multiply_", x.dtype(), y.dtype(), x.shape(), y.shape())) {
    VLOG(5) << "got different data type, run type promotion automatically.";
    LOG_FIRST_N(WARNING, 1)
        << "got different data type, run type promotion "
           "automatically, this may cause data type been changed.";
    auto op_name = phi::TransToFluidOpName("multiply_");
    auto promotion_type = phi::GetPromoteDtype(
        op_name, x.dtype(), y.dtype(), x.shape(), y.shape());

    x = egr::PromoteCastInplace("x", x, promotion_type);
    auto new_y = egr::PromoteCast("y", y, promotion_type);

    return multiply__ad_func(x, new_y);
  }

  // Layout autotune

  if (egr::Controller::Instance().UseLayoutAutoTune()) {
    paddle::small_vector<std::vector<paddle::Tensor>, egr::kSlotSmallVectorSize>
        tensors_vector = {{x}, {y}};

    auto op_name = phi::TransToFluidOpName("multiply_");
    auto transformer = egr::EagerLayoutAutotune(op_name, tensors_vector);
    auto new_x = transformer->TransInTensor("x", x);
    auto new_y = transformer->TransInTensor("y", y);

    VLOG(5) << "Check and Prepare For LAYOUT " << op_name;
    paddle::imperative::LayoutAutotuneGuard guard(
        egr::Controller::Instance().GetCurrentTracer(), false);
    paddle::Tensor& out = multiply__ad_func(new_x, new_y);

    transformer->SetOutTensorLayout(&out);

    // Returns
    return out;
  }

  // Get Input AutoGradMeta
  egr::AutogradMeta* x_autograd_meta =
      egr::EagerUtils::nullable_autograd_meta(x);
  egr::AutogradMeta* y_autograd_meta =
      egr::EagerUtils::nullable_autograd_meta(y);

  VLOG(5) << "Running C++ API: "
          << "multiply_";
  // Before log info

  if (VLOG_IS_ON(3)) {
    const char* INPUT_PRINT_TEMPLATE = "{ Input: [%s]} ";

    std::string input_str = "";
    std::string output_str = "";
    const char* TENSOR_X_TEMPLATE = " \n( x , [%s]), ";
    std::string input_x_str = paddle::string::Sprintf(
        TENSOR_X_TEMPLATE, egr::EagerUtils::TensorStr(x));
    input_str += input_x_str;
    const char* TENSOR_Y_TEMPLATE = " \n( y , [%s]), ";
    std::string input_y_str = paddle::string::Sprintf(
        TENSOR_Y_TEMPLATE, egr::EagerUtils::TensorStr(y));
    input_str += input_y_str;
    VLOG(3) << paddle::string::Sprintf(INPUT_PRINT_TEMPLATE, input_str);
  }

  bool trace_backward = egr::Controller::Instance().HasGrad();
  bool require_any_grad = egr::EagerUtils::ComputeRequireGrad(
      trace_backward, x_autograd_meta, y_autograd_meta);

  // Node Declaration
  std::shared_ptr<MultiplyGradNode> grad_node;
  // Set grad_node before API Call
  if (require_any_grad) {
    phi::RecordEvent node_creation_record_event(
        "multiply node_creation", phi::TracerEventType::OperatorInner, 1);

    grad_node = std::shared_ptr<MultiplyGradNode>(  // NOLINT
        new MultiplyGradNode(1, 2));
    // Set for forward trace
    if (FLAGS_check_nan_inf) {
      grad_node->SetForwardTrace(egr::Controller::Instance().GetPythonStack());
    }
    // SetAttributes if needed
    grad_node->SetAttribute_axis(-1);
    // Set TensorWrappers for Forward Inputs if needed
    auto x_clone = paddle::experimental::assign(x);
    grad_node->SetTensorWrapper_x(x_clone);
    grad_node->SetTensorWrapper_y(y);
  }

  // Forward API Call
  auto& api_result = paddle::experimental::multiply_(x, y);
  // Check NaN and Inf if needed

  if (FLAGS_check_nan_inf) {
    egr::CheckTensorHasNanOrInf("multiply_", api_result);
  }

  // Get Outputs
  auto& out = api_result;

  // Get Output AutoGradMeta
  egr::AutogradMeta* out_autograd_meta = egr::EagerUtils::autograd_meta(&out);
  // Check Inplace if needed

  egr::EagerUtils::CheckInplace(x, x_autograd_meta, require_any_grad);

  // Bump Inplace Version
  x.bump_inplace_version();
  VLOG(3) << "Tensor(" << x.name() << ") uses Inplace Strategy.";

  // Node Creation
  if (require_any_grad) {
    egr::EagerUtils::PassStopGradient(false, out_autograd_meta);
    // SetGradOutMeta & SetEdges
    grad_node->SetGradOutMeta(x, 0);
    grad_node->SetGradOutMeta(y, 1);
    // SetOutRank & SetHistory & SetGradInMeta
    if (out_autograd_meta) {
      egr::EagerUtils::SetOutRankWithSlot(out_autograd_meta, 0);
    }
    if (out_autograd_meta) {
      egr::EagerUtils::SetHistory(out_autograd_meta, grad_node);
    }
    grad_node->SetGradInMeta(out, 0);
    // Set TensorWrappers for Forward Outputs if needed
  }

  VLOG(4) << "Finish AD API: multiply_";
  // LOG IF DEBUG

  if (VLOG_IS_ON(4)) {
    const char* INPUT_PRINT_TEMPLATE = "{ Input: [%s],  \n Output: [%s] } ";

    std::string input_str = "";
    std::string output_str = "";
    const char* TENSOR_X_TEMPLATE = " \n( x , [%s]), ";
    std::string input_x_str = paddle::string::Sprintf(
        TENSOR_X_TEMPLATE, egr::EagerUtils::TensorStr(x));
    input_str += input_x_str;
    const char* TENSOR_Y_TEMPLATE = " \n( y , [%s]), ";
    std::string input_y_str = paddle::string::Sprintf(
        TENSOR_Y_TEMPLATE, egr::EagerUtils::TensorStr(y));
    input_str += input_y_str;
    const char* TENSOR_OUT_TEMPLATE = " \n( out , [%s]), ";
    std::string output_out_str = paddle::string::Sprintf(
        TENSOR_OUT_TEMPLATE, egr::EagerUtils::TensorStr(out));
    output_str += output_out_str;
    VLOG(4) << paddle::string::Sprintf(
        INPUT_PRINT_TEMPLATE, input_str, output_str);
  }

  // Returns
  return out;
}

namespace sparse {

paddle::Tensor multiply_ad_func(const paddle::Tensor& x,
                                const paddle::Tensor& y) {
  FLAGS_tensor_operants_mode = "eager";
  VLOG(3) << "Running AD API: "
          << "multiply";
  // Dygraph Record Event
  phi::RecordEvent dygraph_entrance_record_event(
      "multiply dygraph", phi::TracerEventType::Operator, 1);

  // AMP Logic
  if (egr::Controller::Instance().GetAMPLevel() !=
      paddle::imperative::AmpLevel::O0) {
    VLOG(5) << "Check and Prepare For AMP";
    auto op_name = phi::TransToFluidOpName("multiply");
    paddle::small_vector<std::vector<paddle::Tensor>, egr::kSlotSmallVectorSize>
        amp_tensors_vector = {{x}, {y}};

    auto amp_dst_dtype =
        paddle::imperative::GetAmpDestDtype(op_name, amp_tensors_vector);

    auto new_x =
        paddle::imperative::AmpAutoCast("x", x, amp_dst_dtype, op_name);
    auto new_y =
        paddle::imperative::AmpAutoCast("y", y, amp_dst_dtype, op_name);

    {
      paddle::imperative::AutoCastGuard guard(
          egr::Controller::Instance().GetCurrentAmpAttrs(),
          paddle::imperative::AmpLevel::O0);
      return multiply_ad_func(new_x, new_y);
    }
  }

  // Type promotion Logic
  if (phi::NeedTypePromotion(
          "multiply", x.dtype(), y.dtype(), x.shape(), y.shape())) {
    VLOG(5) << "got different data type, run type promotion automatically.";
    LOG_FIRST_N(WARNING, 1)
        << "got different data type, run type promotion "
           "automatically, this may cause data type been changed.";
    auto op_name = phi::TransToFluidOpName("multiply");
    auto promotion_type = phi::GetPromoteDtype(
        op_name, x.dtype(), y.dtype(), x.shape(), y.shape());

    auto new_x = egr::PromoteCast("x", x, promotion_type);
    auto new_y = egr::PromoteCast("y", y, promotion_type);

    return multiply_ad_func(new_x, new_y);
  }

  // Layout autotune

  if (egr::Controller::Instance().UseLayoutAutoTune()) {
    paddle::small_vector<std::vector<paddle::Tensor>, egr::kSlotSmallVectorSize>
        tensors_vector = {{x}, {y}};

    auto op_name = phi::TransToFluidOpName("multiply");
    auto transformer = egr::EagerLayoutAutotune(op_name, tensors_vector);
    auto new_x = transformer->TransInTensor("x", x);
    auto new_y = transformer->TransInTensor("y", y);

    VLOG(5) << "Check and Prepare For LAYOUT " << op_name;
    paddle::imperative::LayoutAutotuneGuard guard(
        egr::Controller::Instance().GetCurrentTracer(), false);
    paddle::Tensor out = multiply_ad_func(new_x, new_y);

    transformer->SetOutTensorLayout(&out);

    // Returns
    return out;
  }

  // Get Input AutoGradMeta
  egr::AutogradMeta* x_autograd_meta =
      egr::EagerUtils::nullable_autograd_meta(x);
  egr::AutogradMeta* y_autograd_meta =
      egr::EagerUtils::nullable_autograd_meta(y);

  VLOG(5) << "Running C++ API: "
          << "multiply";
  // Before log info

  if (VLOG_IS_ON(3)) {
    const char* INPUT_PRINT_TEMPLATE = "{ Input: [%s]} ";

    std::string input_str = "";
    std::string output_str = "";
    const char* TENSOR_X_TEMPLATE = " \n( x , [%s]), ";
    std::string input_x_str = paddle::string::Sprintf(
        TENSOR_X_TEMPLATE, egr::EagerUtils::TensorStr(x));
    input_str += input_x_str;
    const char* TENSOR_Y_TEMPLATE = " \n( y , [%s]), ";
    std::string input_y_str = paddle::string::Sprintf(
        TENSOR_Y_TEMPLATE, egr::EagerUtils::TensorStr(y));
    input_str += input_y_str;
    VLOG(3) << paddle::string::Sprintf(INPUT_PRINT_TEMPLATE, input_str);
  }
  // Forward API Call
  auto api_result = paddle::experimental::sparse::multiply(x, y);
  // Check NaN and Inf if needed

  if (FLAGS_check_nan_inf) {
    egr::CheckTensorHasNanOrInf("multiply", api_result);
  }

  // Get Outputs
  auto& out = api_result;

  // Get Output AutoGradMeta
  egr::AutogradMeta* out_autograd_meta = egr::EagerUtils::autograd_meta(&out);
  bool trace_backward = egr::Controller::Instance().HasGrad();
  bool require_any_grad = egr::EagerUtils::ComputeRequireGrad(
      trace_backward, x_autograd_meta, y_autograd_meta);

  // Check Inplace if needed

  // Node Creation
  if (require_any_grad) {
    phi::RecordEvent node_creation_record_event(
        "multiply node_creation", phi::TracerEventType::OperatorInner, 1);

    egr::EagerUtils::PassStopGradient(false, out_autograd_meta);

    // Node Construction
    auto grad_node = std::shared_ptr<MultiplyGradNode>(  // NOLINT
        new MultiplyGradNode(1, 2));
    // Set for forward trace
    if (FLAGS_check_nan_inf) {
      grad_node->SetForwardTrace(egr::Controller::Instance().GetPythonStack());
    }
    // SetAttributes if needed

    // Set TensorWrappers for Forward Inputs if needed
    grad_node->SetTensorWrapper_x(x);
    grad_node->SetTensorWrapper_y(y);
    // SetGradOutMeta & SetEdges
    grad_node->SetGradOutMeta(x, 0);
    grad_node->SetGradOutMeta(y, 1);
    // SetOutRank & SetHistory & SetGradInMeta
    if (out_autograd_meta) {
      egr::EagerUtils::SetOutRankWithSlot(out_autograd_meta, 0);
    }
    if (out_autograd_meta) {
      egr::EagerUtils::SetHistory(out_autograd_meta, grad_node);
    }
    grad_node->SetGradInMeta(out, 0);
    // Set TensorWrappers for Forward Outputs if needed
  }

  VLOG(4) << "Finish AD API: multiply";
  // LOG IF DEBUG

  if (VLOG_IS_ON(4)) {
    const char* INPUT_PRINT_TEMPLATE = "{ Input: [%s],  \n Output: [%s] } ";

    std::string input_str = "";
    std::string output_str = "";
    const char* TENSOR_X_TEMPLATE = " \n( x , [%s]), ";
    std::string input_x_str = paddle::string::Sprintf(
        TENSOR_X_TEMPLATE, egr::EagerUtils::TensorStr(x));
    input_str += input_x_str;
    const char* TENSOR_Y_TEMPLATE = " \n( y , [%s]), ";
    std::string input_y_str = paddle::string::Sprintf(
        TENSOR_Y_TEMPLATE, egr::EagerUtils::TensorStr(y));
    input_str += input_y_str;
    const char* TENSOR_OUT_TEMPLATE = " \n( out , [%s]), ";
    std::string output_out_str = paddle::string::Sprintf(
        TENSOR_OUT_TEMPLATE, egr::EagerUtils::TensorStr(out));
    output_str += output_out_str;
    VLOG(4) << paddle::string::Sprintf(
        INPUT_PRINT_TEMPLATE, input_str, output_str);
  }

  // Returns
  return out;
}

}  // namespace sparse
