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

#include "paddle/fluid/eager/amp_utils.h"
#include "paddle/fluid/eager/api/manual/eager_manual/dygraph_forward_api.h"
#include "paddle/fluid/eager/api/manual/eager_manual/nodes/nodes.h"
#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/eager/eager_amp_auto_cast.h"
#include "paddle/fluid/eager/eager_layout_auto_tune.h"
#include "paddle/fluid/eager/nan_inf_utils.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/profiler/event_tracing.h"
#include "paddle/phi/api/include/sparse_api.h"
#include "paddle/phi/core/flags.h"

DECLARE_bool(check_nan_inf);

paddle::Tensor multiply_ad_func(const paddle::Tensor& x,
                                const paddle::Tensor& y) {
  FLAGS_tensor_operants_mode = "eager";
  VLOG(3) << "Running AD API: "
          << "multiply";
  // Dygraph Record Event
  paddle::platform::RecordEvent dygraph_entrance_record_event(
      "multiply dygraph", paddle::platform::TracerEventType::Operator, 1);

  // AMP Logic
  if (egr::Controller::Instance().GetAMPLevel() !=
      paddle::imperative::AmpLevel::O0) {
    VLOG(5) << "Check and Prepare For AMP";
    auto op_name = phi::TransToFluidOpName("multiply");
    paddle::small_vector<std::vector<paddle::Tensor>, egr::kSlotSmallVectorSize>
        amp_tensors_vector = {{x}, {y}};

    auto amp_dst_dtype = egr::GetAmpDestDtype(op_name, amp_tensors_vector);

    auto new_x = egr::EagerAmpAutoCast("x", x, amp_dst_dtype, op_name);
    auto new_y = egr::EagerAmpAutoCast("y", y, amp_dst_dtype, op_name);

    {
      paddle::imperative::AutoCastGuard guard(
          egr::Controller::Instance().GetCurrentTracer(),
          paddle::imperative::AmpLevel::O0);
      return multiply_ad_func(new_x, new_y);
    }
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
    paddle::platform::RecordEvent node_creation_record_event(
        "multiply node_creation",
        paddle::platform::TracerEventType::OperatorInner,
        1);

    egr::EagerUtils::PassStopGradient(false, out_autograd_meta);

    // Node Construction
    auto grad_node =
        std::shared_ptr<MultiplyGradNode>(new MultiplyGradNode(1, 2));

    // SetAttributes if needed
    grad_node->SetAttributeaxis(-1);
    // Set TensorWrappers for Forward Inputs if needed
    if (paddle::platform::is_gpu_place(x.place())) {
      if (!x_autograd_meta->StopGradient() &&
          !y_autograd_meta->StopGradient()) {
        grad_node->SetTensorWrapperx(x);
        grad_node->SetTensorWrappery(y);
      } else if (x_autograd_meta->StopGradient() &&
                 !y_autograd_meta->StopGradient()) {
        grad_node->SetTensorWrapperx(x);
        grad_node->SetTensorWrapperNoNeedBuffery(y);
      } else if (!x_autograd_meta->StopGradient() &&
                 y_autograd_meta->StopGradient()) {
        grad_node->SetTensorWrapperNoNeedBufferx(x);
        grad_node->SetTensorWrappery(y);
      } else {
        grad_node->SetTensorWrapperNoNeedBufferx(x);
        grad_node->SetTensorWrapperNoNeedBuffery(y);
      }
    } else {
      grad_node->SetTensorWrapperx(x);
      grad_node->SetTensorWrappery(y);
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
  paddle::platform::RecordEvent dygraph_entrance_record_event(
      "multiply_ dygraph", paddle::platform::TracerEventType::Operator, 1);

  // AMP Logic

  VLOG(5)
      << " No AMP for multiply__ad_func because it is a inplace or cast api. ";
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
  bool trace_backward = egr::Controller::Instance().HasGrad();
  bool require_any_grad = egr::EagerUtils::ComputeRequireGrad(
      trace_backward, x_autograd_meta, y_autograd_meta);

  // Check Inplace if needed

  egr::EagerUtils::CheckInplace(x, x_autograd_meta, require_any_grad);

  // Bump Inplace Version
  x.bump_inplace_version();
  VLOG(3) << "Tensor(" << x.name() << ") uses Inplace Strategy.";

  // Node Creation
  if (require_any_grad) {
    paddle::platform::RecordEvent node_creation_record_event(
        "multiply node_creation",
        paddle::platform::TracerEventType::OperatorInner,
        1);

    egr::EagerUtils::PassStopGradient(false, out_autograd_meta);

    // Node Construction
    auto grad_node =
        std::shared_ptr<MultiplyGradNode>(new MultiplyGradNode(1, 2));

    // SetAttributes if needed
    grad_node->SetAttributeaxis(-1);
    // Set TensorWrappers for Forward Inputs if needed
    grad_node->SetTensorWrapperx(x);
    grad_node->SetTensorWrappery(y);
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
  paddle::platform::RecordEvent dygraph_entrance_record_event(
      "multiply dygraph", paddle::platform::TracerEventType::Operator, 1);

  // AMP Logic
  if (egr::Controller::Instance().GetAMPLevel() !=
      paddle::imperative::AmpLevel::O0) {
    VLOG(5) << "Check and Prepare For AMP";
    auto op_name = phi::TransToFluidOpName("multiply");
    paddle::small_vector<std::vector<paddle::Tensor>, egr::kSlotSmallVectorSize>
        amp_tensors_vector = {{x}, {y}};

    auto amp_dst_dtype = egr::GetAmpDestDtype(op_name, amp_tensors_vector);

    auto new_x = egr::EagerAmpAutoCast("x", x, amp_dst_dtype, op_name);
    auto new_y = egr::EagerAmpAutoCast("y", y, amp_dst_dtype, op_name);

    {
      paddle::imperative::AutoCastGuard guard(
          egr::Controller::Instance().GetCurrentTracer(),
          paddle::imperative::AmpLevel::O0);
      return multiply_ad_func(new_x, new_y);
    }
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
    paddle::platform::RecordEvent node_creation_record_event(
        "multiply node_creation",
        paddle::platform::TracerEventType::OperatorInner,
        1);

    egr::EagerUtils::PassStopGradient(false, out_autograd_meta);

    // Node Construction
    auto grad_node =
        std::shared_ptr<MultiplyGradNode>(new MultiplyGradNode(1, 2));
    // SetAttributes if needed

    // Set TensorWrappers for Forward Inputs if needed
    grad_node->SetTensorWrapperx(x);
    grad_node->SetTensorWrappery(y);
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
