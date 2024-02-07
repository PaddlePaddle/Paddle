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
#include "paddle/fluid/eager/amp_utils.h"
#include "paddle/fluid/eager/api/generated/eager_generated/backwards/nodes.h"
#include "paddle/fluid/eager/api/manual/eager_manual/dygraph_forward_api.h"
#include "paddle/fluid/eager/api/manual/eager_manual/nodes/nodes.h"
#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/eager/eager_amp_auto_cast.h"
#include "paddle/fluid/eager/eager_layout_auto_tune.h"
#include "paddle/fluid/eager/nan_inf_utils.h"
#include "paddle/fluid/platform/profiler/event_tracing.h"
#include "paddle/phi/api/lib/data_transform.h"

COMMON_DECLARE_bool(check_nan_inf);

TEST_API paddle::Tensor cast_ad_func(const paddle::Tensor& x,
                                     phi::DataType dtype) {
  FLAGS_tensor_operants_mode = "eager";
  VLOG(3) << "Running AD API: "
          << "cast";
  // Dygraph Record Event
  paddle::platform::RecordEvent dygraph_entrance_record_event(
      "cast dygraph", paddle::platform::TracerEventType::Operator, 1);

  if (x.dtype() == dtype) {
    return x;
  }

  // AMP Logic

  VLOG(5) << " No AMP for cast_ad_func because it is a inplace or cast api. ";
  // Type promotion Logic

  VLOG(5) << " No Type Promotion for cast_ad_func api. ";
  // Layout autotune

  if (egr::Controller::Instance().UseLayoutAutoTune()) {
    paddle::small_vector<std::vector<paddle::Tensor>, egr::kSlotSmallVectorSize>
        tensors_vector = {{x}};

    auto op_name = phi::TransToFluidOpName("cast");
    auto transformer = egr::EagerLayoutAutotune(op_name, tensors_vector);
    auto new_x = transformer->TransInTensor("x", x);

    VLOG(5) << "Check and Prepare For LAYOUT " << op_name;
    paddle::imperative::LayoutAutotuneGuard guard(
        egr::Controller::Instance().GetCurrentTracer(), false);
    paddle::Tensor out = cast_ad_func(new_x, dtype);

    transformer->SetOutTensorLayout(&out);

    // Returns
    return out;
  }

  // Get Input AutoGradMeta
  egr::AutogradMeta* x_autograd_meta =
      egr::EagerUtils::nullable_autograd_meta(x);

  VLOG(5) << "Running C++ API: "
          << "cast";
  // Before log info

  if (VLOG_IS_ON(3)) {
    const char* INPUT_PRINT_TEMPLATE = "{ Input: [%s]} ";

    std::string input_str = "";
    std::string output_str = "";
    const char* TENSOR_X_TEMPLATE = " \n( x , [%s]), ";
    std::string input_x_str = paddle::string::Sprintf(
        TENSOR_X_TEMPLATE, egr::EagerUtils::TensorStr(x));
    input_str += input_x_str;
    VLOG(3) << paddle::string::Sprintf(INPUT_PRINT_TEMPLATE, input_str);
  }

  bool trace_backward = egr::Controller::Instance().HasGrad();
  bool require_any_grad =
      egr::EagerUtils::ComputeRequireGrad(trace_backward, x_autograd_meta);

  // Node Declaration
  std::shared_ptr<CastGradNode> grad_node;

  // Pre contiguous tensor in not strided op, if 1)require_any_grad=true; 2)
  // need wrapper to backward; 3) not contiguous
  const auto& x_tmp =
      (require_any_grad && x.is_dense_tensor() &&
       !std::dynamic_pointer_cast<phi::DenseTensor>(x.impl())
            ->meta()
            .is_contiguous())
          ? paddle::Tensor(std::make_shared<phi::DenseTensor>(std::move(
                               paddle::experimental::Trans2Contiguous(*(
                                   std::dynamic_pointer_cast<phi::DenseTensor>(
                                       x.impl()))))),
                           x.mutable_autograd_meta())
          : x;

  // Set grad_node before API Call
  if (require_any_grad) {
    paddle::platform::RecordEvent node_creation_record_event(
        "cast node_creation",
        paddle::platform::TracerEventType::OperatorInner,
        1);

    // Node Construction
    grad_node =
        std::shared_ptr<CastGradNode>(new CastGradNode(1, 1));  // NOLINT
    // Set for forward trace
    if (FLAGS_check_nan_inf) {
      grad_node->SetForwardTrace(egr::Controller::Instance().GetPythonStack());
    }
    // SetAttributes if needed

    // Set TensorWrappers for Forward Inputs if needed
    grad_node->SetTensorWrapperx(x_tmp);
  }

  // Forward API Call
  auto api_result = paddle::experimental::cast(x_tmp, dtype);
  // Log memory infomation
  paddle::memory::LogDeviceMemoryStats(
      egr::Controller::Instance().GetExpectedPlace(), "cast");
  // Check NaN and Inf if needed

  if (FLAGS_check_nan_inf) {
    egr::CheckTensorHasNanOrInf("cast", api_result);
  }

  // Get Outputs
  auto& out = api_result;

  // Get Output AutoGradMeta
  egr::AutogradMeta* out_autograd_meta = egr::EagerUtils::autograd_meta(&out);
  // Check Inplace if needed

  // Set grad_node after API call
  if (require_any_grad) {
    egr::EagerUtils::PassStopGradient(false, out_autograd_meta);

    // SetGradOutMeta & SetEdges
    grad_node->SetGradOutMeta(x, 0);
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

  VLOG(4) << "Finish AD API: cast";
  // LOG IF DEBUG

  if (VLOG_IS_ON(4)) {
    const char* INPUT_PRINT_TEMPLATE = "{ Input: [%s],  \n Output: [%s] } ";

    std::string input_str = "";
    std::string output_str = "";
    const char* TENSOR_X_TEMPLATE = " \n( x , [%s]), ";
    std::string input_x_str = paddle::string::Sprintf(
        TENSOR_X_TEMPLATE, egr::EagerUtils::TensorStr(x));
    input_str += input_x_str;
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

TEST_API paddle::Tensor& cast__ad_func(paddle::Tensor& x,  // NOLINT
                                       phi::DataType dtype) {
  FLAGS_tensor_operants_mode = "eager";
  VLOG(3) << "Running AD API: "
          << "cast_";
  // Dygraph Record Event
  paddle::platform::RecordEvent dygraph_entrance_record_event(
      "cast_ dygraph", paddle::platform::TracerEventType::Operator, 1);

  // AMP Logic

  VLOG(5) << " No AMP for cast__ad_func because it is a inplace or cast api. ";
  // Type promotion Logic

  VLOG(5) << " No Type Promotion for cast__ad_func api. ";
  // Layout autotune

  if (egr::Controller::Instance().UseLayoutAutoTune()) {
    paddle::small_vector<std::vector<paddle::Tensor>, egr::kSlotSmallVectorSize>
        tensors_vector = {{x}};

    auto op_name = phi::TransToFluidOpName("cast_");
    auto transformer = egr::EagerLayoutAutotune(op_name, tensors_vector);
    auto new_x = transformer->TransInTensor("x", x);

    VLOG(5) << "Check and Prepare For LAYOUT " << op_name;
    paddle::imperative::LayoutAutotuneGuard guard(
        egr::Controller::Instance().GetCurrentTracer(), false);
    paddle::Tensor& out = cast__ad_func(new_x, dtype);

    transformer->SetOutTensorLayout(&out);

    // Returns
    return out;
  }

  // Get Input AutoGradMeta
  egr::AutogradMeta* x_autograd_meta =
      egr::EagerUtils::nullable_autograd_meta(x);

  VLOG(5) << "Running C++ API: "
          << "cast_";
  // Before log info

  if (VLOG_IS_ON(3)) {
    const char* INPUT_PRINT_TEMPLATE = "{ Input: [%s]} ";

    std::string input_str = "";
    std::string output_str = "";
    const char* TENSOR_X_TEMPLATE = " \n( x , [%s]), ";
    std::string input_x_str = paddle::string::Sprintf(
        TENSOR_X_TEMPLATE, egr::EagerUtils::TensorStr(x));
    input_str += input_x_str;
    VLOG(3) << paddle::string::Sprintf(INPUT_PRINT_TEMPLATE, input_str);
  }

  bool trace_backward = egr::Controller::Instance().HasGrad();
  bool require_any_grad =
      egr::EagerUtils::ComputeRequireGrad(trace_backward, x_autograd_meta);

  // Node Declaration
  std::shared_ptr<CastGradNode> grad_node;

  // Pre contiguous tensor in not strided op, if 1)require_any_grad=true; 2)
  // need wrapper to backward; 3) not contiguous

  // Set grad_node before API Call
  if (require_any_grad) {
    paddle::platform::RecordEvent node_creation_record_event(
        "cast node_creation",
        paddle::platform::TracerEventType::OperatorInner,
        1);

    // Node Construction
    grad_node =
        std::shared_ptr<CastGradNode>(new CastGradNode(1, 1));  // NOLINT
    // Set for forward trace
    if (FLAGS_check_nan_inf) {
      grad_node->SetForwardTrace(egr::Controller::Instance().GetPythonStack());
    }
    // SetAttributes if needed

    // Set TensorWrappers for Forward Inputs if needed
    auto x_clone = paddle::experimental::assign(x);
    grad_node->SetTensorWrapperx(x_clone);
  }

  // Forward API Call
  auto& api_result = paddle::experimental::cast_(x, dtype);
  // Log memory infomation
  paddle::memory::LogDeviceMemoryStats(
      egr::Controller::Instance().GetExpectedPlace(), "cast_");
  // Check NaN and Inf if needed

  if (FLAGS_check_nan_inf) {
    egr::CheckTensorHasNanOrInf("cast_", api_result);
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

  // Set grad_node after API call
  if (require_any_grad) {
    egr::EagerUtils::PassStopGradient(false, out_autograd_meta);

    // SetGradOutMeta & SetEdges
    grad_node->SetGradOutMeta(x, 0);
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

  VLOG(4) << "Finish AD API: cast_";
  // LOG IF DEBUG

  if (VLOG_IS_ON(4)) {
    const char* INPUT_PRINT_TEMPLATE = "{ Input: [%s],  \n Output: [%s] } ";

    std::string input_str = "";
    std::string output_str = "";
    const char* TENSOR_X_TEMPLATE = " \n( x , [%s]), ";
    std::string input_x_str = paddle::string::Sprintf(
        TENSOR_X_TEMPLATE, egr::EagerUtils::TensorStr(x));
    input_str += input_x_str;
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
TEST_API paddle::Tensor cast_ad_func(const paddle::Tensor& x,
                                     phi::DataType index_dtype,
                                     phi::DataType value_dtype) {
  FLAGS_tensor_operants_mode = "eager";
  VLOG(3) << "Running AD API: "
          << "cast";
  // Dygraph Record Event
  paddle::platform::RecordEvent dygraph_entrance_record_event(
      "cast dygraph", paddle::platform::TracerEventType::Operator, 1);

  // AMP Logic

  VLOG(5) << " No AMP for cast_ad_func because it is a inplace or cast api. ";
  // Type promotion Logic

  VLOG(5) << " No Type Promotion for cast_ad_func api. ";
  // Layout autotune

  if (egr::Controller::Instance().UseLayoutAutoTune()) {
    paddle::small_vector<std::vector<paddle::Tensor>, egr::kSlotSmallVectorSize>
        tensors_vector = {{x}};

    auto op_name = phi::TransToFluidOpName("cast");
    auto transformer = egr::EagerLayoutAutotune(op_name, tensors_vector);
    auto new_x = transformer->TransInTensor("x", x);

    VLOG(5) << "Check and Prepare For LAYOUT " << op_name;
    paddle::imperative::LayoutAutotuneGuard guard(
        egr::Controller::Instance().GetCurrentTracer(), false);
    paddle::Tensor out = cast_ad_func(new_x, index_dtype, value_dtype);

    transformer->SetOutTensorLayout(&out);

    // Returns
    return out;
  }

  // Get Input AutoGradMeta
  egr::AutogradMeta* x_autograd_meta =
      egr::EagerUtils::nullable_autograd_meta(x);

  VLOG(5) << "Running C++ API: "
          << "cast";
  // Before log info

  if (VLOG_IS_ON(3)) {
    const char* INPUT_PRINT_TEMPLATE = "{ Input: [%s]} ";

    std::string input_str = "";
    std::string output_str = "";
    const char* TENSOR_X_TEMPLATE = " \n( x , [%s]), ";
    std::string input_x_str = paddle::string::Sprintf(
        TENSOR_X_TEMPLATE, egr::EagerUtils::TensorStr(x));
    input_str += input_x_str;
    VLOG(3) << paddle::string::Sprintf(INPUT_PRINT_TEMPLATE, input_str);
  }

  bool trace_backward = egr::Controller::Instance().HasGrad();
  bool require_any_grad =
      egr::EagerUtils::ComputeRequireGrad(trace_backward, x_autograd_meta);

  // Node Declaration
  std::shared_ptr<CastGradNode> grad_node;

  // Pre contiguous tensor in not strided op, if 1)require_any_grad=true; 2)
  // need wrapper to backward; 3) not contiguous
  const auto& x_tmp =
      (require_any_grad && x.is_dense_tensor() &&
       !std::dynamic_pointer_cast<phi::DenseTensor>(x.impl())
            ->meta()
            .is_contiguous())
          ? paddle::Tensor(std::make_shared<phi::DenseTensor>(std::move(
                               paddle::experimental::Trans2Contiguous(*(
                                   std::dynamic_pointer_cast<phi::DenseTensor>(
                                       x.impl()))))),
                           x.mutable_autograd_meta())
          : x;

  // Set grad_node before API Call
  if (require_any_grad) {
    paddle::platform::RecordEvent node_creation_record_event(
        "cast node_creation",
        paddle::platform::TracerEventType::OperatorInner,
        1);

    // Node Construction
    grad_node =
        std::shared_ptr<CastGradNode>(new CastGradNode(1, 1));  // NOLINT
    // Set for forward trace
    if (FLAGS_check_nan_inf) {
      grad_node->SetForwardTrace(egr::Controller::Instance().GetPythonStack());
    }
    // SetAttributes if needed
    grad_node->SetAttributevalue_dtype(value_dtype);
    // Set TensorWrappers for Forward Inputs if needed
    grad_node->SetTensorWrapperx(x_tmp);
  }

  // Forward API Call
  auto api_result =
      paddle::experimental::sparse::cast(x_tmp, index_dtype, value_dtype);
  // Log memory infomation
  paddle::memory::LogDeviceMemoryStats(
      egr::Controller::Instance().GetExpectedPlace(), "cast");
  // Check NaN and Inf if needed

  if (FLAGS_check_nan_inf) {
    egr::CheckTensorHasNanOrInf("cast", api_result);
  }

  // Get Outputs
  auto& out = api_result;

  // Get Output AutoGradMeta
  egr::AutogradMeta* out_autograd_meta = egr::EagerUtils::autograd_meta(&out);
  // Check Inplace if needed

  // Set grad_node after API call
  if (require_any_grad) {
    egr::EagerUtils::PassStopGradient(false, out_autograd_meta);

    // SetGradOutMeta & SetEdges
    grad_node->SetGradOutMeta(x, 0);
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

  VLOG(4) << "Finish AD API: cast";
  // LOG IF DEBUG

  if (VLOG_IS_ON(4)) {
    const char* INPUT_PRINT_TEMPLATE = "{ Input: [%s],  \n Output: [%s] } ";

    std::string input_str = "";
    std::string output_str = "";
    const char* TENSOR_X_TEMPLATE = " \n( x , [%s]), ";
    std::string input_x_str = paddle::string::Sprintf(
        TENSOR_X_TEMPLATE, egr::EagerUtils::TensorStr(x));
    input_str += input_x_str;
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
