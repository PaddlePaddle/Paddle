// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/platform/profiler/event_tracing.h"
#include "paddle/phi/api/include/sparse_api.h"

#pragma GCC diagnostic ignored "-Wunused-variable"
DECLARE_bool(check_nan_inf);
DECLARE_string(tensor_operants_mode);

std::tuple<paddle::Tensor,
           paddle::Tensor&,
           paddle::Tensor&,
           paddle::Tensor,
           paddle::Tensor,
           paddle::Tensor>
sync_batch_norm__ad_func(const paddle::Tensor& x,
                         const paddle::Tensor& scale,
                         const paddle::Tensor& bias,
                         paddle::Tensor& mean,      // NOLINT
                         paddle::Tensor& variance,  // NOLINT
                         float momentum,
                         float epsilon,
                         std::string data_layout,
                         bool is_test,
                         bool use_global_stats,
                         bool trainable_statistics,
                         bool fuse_with_relu) {
  VLOG(3) << "Running AD API: "
          << "sync_batch_norm_";
  // Dygraph Record Event
  paddle::platform::RecordEvent dygraph_entrance_record_event(
      "sync_batch_norm_ dygraph",
      paddle::platform::TracerEventType::Operator,
      1);

  // AMP Logic

  VLOG(5) << " No AMP for sync_batch_norm__ad_func because it is a inplace or "
             "cast api. ";
  // Layout autotune

  if (egr::Controller::Instance().UseLayoutAutoTune()) {
    paddle::small_vector<std::vector<paddle::Tensor>, egr::kSlotSmallVectorSize>
        tensors_vector = {{x}, {scale}, {bias}, {mean}, {variance}};

    auto op_name = phi::TransToFluidOpName("sync_batch_norm_");
    auto transformer = egr::EagerLayoutAutotune<std::string>(
        op_name, tensors_vector, &data_layout);
    auto new_x = transformer->TransInTensor("x", x);
    auto new_scale = transformer->TransInTensor("scale", scale);
    auto new_bias = transformer->TransInTensor("bias", bias);
    auto new_mean = transformer->TransInTensor("mean", mean);
    auto new_variance = transformer->TransInTensor("variance", variance);

    VLOG(5) << "Check and Prepare For LAYOUT " << op_name;
    paddle::imperative::LayoutAutotuneGuard guard(
        egr::Controller::Instance().GetCurrentTracer(), false);
    std::tuple<paddle::Tensor,
               paddle::Tensor&,
               paddle::Tensor&,
               paddle::Tensor,
               paddle::Tensor,
               paddle::Tensor>
        api_result = sync_batch_norm__ad_func(new_x,
                                              new_scale,
                                              new_bias,
                                              new_mean,
                                              new_variance,
                                              momentum,
                                              epsilon,
                                              data_layout,
                                              is_test,
                                              use_global_stats,
                                              trainable_statistics,
                                              fuse_with_relu);

    auto& out = std::get<0>(api_result);
    transformer->SetOutTensorLayout(&out);
    auto& mean_out = std::get<1>(api_result);
    transformer->SetOutTensorLayout(&mean_out);
    auto& variance_out = std::get<2>(api_result);
    transformer->SetOutTensorLayout(&variance_out);
    auto& saved_mean = std::get<3>(api_result);
    transformer->SetOutTensorLayout(&saved_mean);
    auto& saved_variance = std::get<4>(api_result);
    transformer->SetOutTensorLayout(&saved_variance);
    auto& reserve_space = std::get<5>(api_result);
    transformer->SetOutTensorLayout(&reserve_space);

    // Returns
    return std::tuple<paddle::Tensor,
                      paddle::Tensor&,
                      paddle::Tensor&,
                      paddle::Tensor,
                      paddle::Tensor,
                      paddle::Tensor>{
        out, mean_out, variance_out, saved_mean, saved_variance, reserve_space};
  }

  // Get Input AutoGradMeta
  egr::AutogradMeta* x_autograd_meta =
      egr::EagerUtils::nullable_autograd_meta(x);
  egr::AutogradMeta* scale_autograd_meta =
      egr::EagerUtils::nullable_autograd_meta(scale);
  egr::AutogradMeta* bias_autograd_meta =
      egr::EagerUtils::nullable_autograd_meta(bias);
  egr::AutogradMeta* mean_autograd_meta =
      egr::EagerUtils::nullable_autograd_meta(mean);
  egr::AutogradMeta* variance_autograd_meta =
      egr::EagerUtils::nullable_autograd_meta(variance);

  VLOG(5) << "Running C++ API: "
          << "sync_batch_norm_";
  // Before log info

  if (VLOG_IS_ON(3)) {
    const char* INPUT_PRINT_TEMPLATE = "{ Input: [%s]} ";

    std::string input_str = "";
    std::string output_str = "";
    const char* TENSOR_X_TEMPLATE = "(x, [%s]), ";
    std::string input_x_str = paddle::string::Sprintf(
        TENSOR_X_TEMPLATE, egr::EagerUtils::TensorStr(x));
    input_str += input_x_str;
    const char* TENSOR_SCALE_TEMPLATE = "(scale, [%s]), ";
    std::string input_scale_str = paddle::string::Sprintf(
        TENSOR_SCALE_TEMPLATE, egr::EagerUtils::TensorStr(scale));
    input_str += input_scale_str;
    const char* TENSOR_BIAS_TEMPLATE = "(bias, [%s]), ";
    std::string input_bias_str = paddle::string::Sprintf(
        TENSOR_BIAS_TEMPLATE, egr::EagerUtils::TensorStr(bias));
    input_str += input_bias_str;
    const char* TENSOR_MEAN_TEMPLATE = "(mean, [%s]), ";
    std::string input_mean_str = paddle::string::Sprintf(
        TENSOR_MEAN_TEMPLATE, egr::EagerUtils::TensorStr(mean));
    input_str += input_mean_str;
    const char* TENSOR_VARIANCE_TEMPLATE = "(variance, [%s]), ";
    std::string input_variance_str = paddle::string::Sprintf(
        TENSOR_VARIANCE_TEMPLATE, egr::EagerUtils::TensorStr(variance));
    input_str += input_variance_str;
    VLOG(3) << paddle::string::Sprintf(INPUT_PRINT_TEMPLATE, input_str);
  }

  // Forward API Call
  auto api_result = paddle::experimental::sync_batch_norm_(x,
                                                           scale,
                                                           bias,
                                                           mean,
                                                           variance,
                                                           momentum,
                                                           epsilon,
                                                           data_layout,
                                                           is_test,
                                                           use_global_stats,
                                                           trainable_statistics,
                                                           fuse_with_relu);
  // Check NaN and Inf if needed
  if (FLAGS_check_nan_inf) {
    egr::CheckTensorHasNanOrInf("sync_batch_norm_", api_result);
  }

  // Get Outputs
  auto& out = std::get<0>(api_result);
  auto& mean_out = std::get<1>(api_result);
  auto& variance_out = std::get<2>(api_result);
  auto& saved_mean = std::get<3>(api_result);
  auto& saved_variance = std::get<4>(api_result);
  auto& reserve_space = std::get<5>(api_result);

  // Get Output AutoGradMeta
  egr::AutogradMeta* out_autograd_meta = egr::EagerUtils::autograd_meta(&out);
  egr::AutogradMeta* mean_out_autograd_meta =
      egr::EagerUtils::autograd_meta(&mean_out);
  egr::AutogradMeta* variance_out_autograd_meta =
      egr::EagerUtils::autograd_meta(&variance_out);
  egr::AutogradMeta* saved_mean_autograd_meta =
      egr::EagerUtils::autograd_meta(&saved_mean);
  egr::AutogradMeta* saved_variance_autograd_meta =
      egr::EagerUtils::autograd_meta(&saved_variance);
  egr::AutogradMeta* reserve_space_autograd_meta =
      egr::EagerUtils::autograd_meta(&reserve_space);
  bool trace_backward = egr::Controller::Instance().HasGrad();
  bool require_any_grad =
      egr::EagerUtils::ComputeRequireGrad(trace_backward,
                                          x_autograd_meta,
                                          scale_autograd_meta,
                                          bias_autograd_meta,
                                          mean_autograd_meta,
                                          variance_autograd_meta);

  // Check Inplace if needed

  egr::EagerUtils::CheckInplace(mean, mean_autograd_meta, require_any_grad);

  egr::EagerUtils::CheckInplace(
      variance, variance_autograd_meta, require_any_grad);

  // Bump Inplace Version
  mean.bump_inplace_version();
  VLOG(3) << "Tensor(" << mean.name() << ") uses Inplace Strategy.";

  // Bump Inplace Version
  variance.bump_inplace_version();
  VLOG(3) << "Tensor(" << variance.name() << ") uses Inplace Strategy.";

  // Node Creation
  if (require_any_grad) {
    paddle::platform::RecordEvent node_creation_record_event(
        "sync_batch_norm_ node_creation",
        paddle::platform::TracerEventType::OperatorInner,
        1);

    egr::EagerUtils::PassStopGradient(false,
                                      out_autograd_meta,
                                      mean_out_autograd_meta,
                                      variance_out_autograd_meta,
                                      saved_mean_autograd_meta,
                                      saved_variance_autograd_meta,
                                      reserve_space_autograd_meta);

    // Node Construction
    auto grad_node =
        std::shared_ptr<SyncBatchNormGradNode>(new SyncBatchNormGradNode(6, 5));
    egr::Controller::Instance().PushBackForceSequentialNodes(grad_node.get());
    // SetAttributes if needed
    grad_node->SetAttributemomentum(momentum);
    grad_node->SetAttributeepsilon(epsilon);
    grad_node->SetAttributedata_layout(data_layout);
    grad_node->SetAttributeis_test(is_test);
    grad_node->SetAttributeuse_global_stats(use_global_stats);
    grad_node->SetAttributetrainable_statistics(trainable_statistics);
    grad_node->SetAttributefuse_with_relu(fuse_with_relu);
    // Set TensorWrappers for Forward Inputs if needed
    grad_node->SetTensorWrapperx(x);
    grad_node->SetTensorWrapperscale(scale);
    grad_node->SetTensorWrapperbias(bias);
    // SetGradOutMeta & SetEdges
    grad_node->SetGradOutMeta(x, 0);
    grad_node->SetGradOutMeta(scale, 1);
    grad_node->SetGradOutMeta(bias, 2);
    // SetOutRank & SetHistory & SetGradInMeta & RetainGrad
    if (out_autograd_meta) {
      egr::EagerUtils::SetOutRankWithSlot(out_autograd_meta, 0);
    }
    if (mean_out_autograd_meta) {
      egr::EagerUtils::SetOutRankWithSlot(mean_out_autograd_meta, 1);
    }
    if (variance_out_autograd_meta) {
      egr::EagerUtils::SetOutRankWithSlot(variance_out_autograd_meta, 2);
    }
    if (saved_mean_autograd_meta) {
      egr::EagerUtils::SetOutRankWithSlot(saved_mean_autograd_meta, 3);
    }
    if (saved_variance_autograd_meta) {
      egr::EagerUtils::SetOutRankWithSlot(saved_variance_autograd_meta, 4);
    }
    if (reserve_space_autograd_meta) {
      egr::EagerUtils::SetOutRankWithSlot(reserve_space_autograd_meta, 5);
    }
    if (out_autograd_meta) {
      egr::EagerUtils::SetHistory(out_autograd_meta, grad_node);
    }
    if (mean_out_autograd_meta) {
      egr::EagerUtils::SetHistory(mean_out_autograd_meta, grad_node);
    }
    if (variance_out_autograd_meta) {
      egr::EagerUtils::SetHistory(variance_out_autograd_meta, grad_node);
    }
    if (saved_mean_autograd_meta) {
      egr::EagerUtils::SetHistory(saved_mean_autograd_meta, grad_node);
    }
    if (saved_variance_autograd_meta) {
      egr::EagerUtils::SetHistory(saved_variance_autograd_meta, grad_node);
    }
    if (reserve_space_autograd_meta) {
      egr::EagerUtils::SetHistory(reserve_space_autograd_meta, grad_node);
    }
    grad_node->SetGradInMeta(out, 0);
    grad_node->SetGradInMeta(mean_out, 1);
    grad_node->SetGradInMeta(variance_out, 2);
    grad_node->SetGradInMeta(saved_mean, 3);
    grad_node->SetGradInMeta(saved_variance, 4);
    grad_node->SetGradInMeta(reserve_space, 5);
    egr::EagerUtils::CheckAndRetainGrad(out);
    egr::EagerUtils::CheckAndRetainGrad(mean_out);
    egr::EagerUtils::CheckAndRetainGrad(variance_out);
    egr::EagerUtils::CheckAndRetainGrad(saved_mean);
    egr::EagerUtils::CheckAndRetainGrad(saved_variance);
    egr::EagerUtils::CheckAndRetainGrad(reserve_space);
    // Set TensorWrappers for Forward Outputs if needed
    grad_node->SetTensorWrappersaved_mean(saved_mean);
    grad_node->SetTensorWrappersaved_variance(saved_variance);
    grad_node->SetTensorWrapperreserve_space(reserve_space);
  }

  VLOG(4) << "Finish AD API: sync_batch_norm_";
  // LOG IF DEBUG

  if (VLOG_IS_ON(4)) {
    const char* INPUT_PRINT_TEMPLATE = "{ Input: [%s],  Output: [%s] } ";

    std::string input_str = "";
    std::string output_str = "";
    const char* TENSOR_X_TEMPLATE = "(x, [%s]), ";
    std::string input_x_str = paddle::string::Sprintf(
        TENSOR_X_TEMPLATE, egr::EagerUtils::TensorStr(x));
    input_str += input_x_str;
    const char* TENSOR_SCALE_TEMPLATE = "(scale, [%s]), ";
    std::string input_scale_str = paddle::string::Sprintf(
        TENSOR_SCALE_TEMPLATE, egr::EagerUtils::TensorStr(scale));
    input_str += input_scale_str;
    const char* TENSOR_BIAS_TEMPLATE = "(bias, [%s]), ";
    std::string input_bias_str = paddle::string::Sprintf(
        TENSOR_BIAS_TEMPLATE, egr::EagerUtils::TensorStr(bias));
    input_str += input_bias_str;
    const char* TENSOR_MEAN_TEMPLATE = "(mean, [%s]), ";
    std::string input_mean_str = paddle::string::Sprintf(
        TENSOR_MEAN_TEMPLATE, egr::EagerUtils::TensorStr(mean));
    input_str += input_mean_str;
    const char* TENSOR_VARIANCE_TEMPLATE = "(variance, [%s]), ";
    std::string input_variance_str = paddle::string::Sprintf(
        TENSOR_VARIANCE_TEMPLATE, egr::EagerUtils::TensorStr(variance));
    input_str += input_variance_str;
    const char* TENSOR_OUT_TEMPLATE = "(out, [%s]), ";
    std::string output_out_str = paddle::string::Sprintf(
        TENSOR_OUT_TEMPLATE, egr::EagerUtils::TensorStr(out));
    output_str += output_out_str;
    const char* TENSOR_MEAN_OUT_TEMPLATE = "(mean_out, [%s]), ";
    std::string output_mean_out_str = paddle::string::Sprintf(
        TENSOR_MEAN_OUT_TEMPLATE, egr::EagerUtils::TensorStr(mean_out));
    output_str += output_mean_out_str;
    const char* TENSOR_VARIANCE_OUT_TEMPLATE = "(variance_out, [%s]), ";
    std::string output_variance_out_str = paddle::string::Sprintf(
        TENSOR_VARIANCE_OUT_TEMPLATE, egr::EagerUtils::TensorStr(variance_out));
    output_str += output_variance_out_str;
    const char* TENSOR_SAVED_MEAN_TEMPLATE = "(saved_mean, [%s]), ";
    std::string output_saved_mean_str = paddle::string::Sprintf(
        TENSOR_SAVED_MEAN_TEMPLATE, egr::EagerUtils::TensorStr(saved_mean));
    output_str += output_saved_mean_str;
    const char* TENSOR_SAVED_VARIANCE_TEMPLATE = "(saved_variance, [%s]), ";
    std::string output_saved_variance_str =
        paddle::string::Sprintf(TENSOR_SAVED_VARIANCE_TEMPLATE,
                                egr::EagerUtils::TensorStr(saved_variance));
    output_str += output_saved_variance_str;
    const char* TENSOR_RESERVE_SPACE_TEMPLATE = "(reserve_space, [%s]), ";
    std::string output_reserve_space_str =
        paddle::string::Sprintf(TENSOR_RESERVE_SPACE_TEMPLATE,
                                egr::EagerUtils::TensorStr(reserve_space));
    output_str += output_reserve_space_str;
    VLOG(4) << paddle::string::Sprintf(
        INPUT_PRINT_TEMPLATE, input_str, output_str);
  }

  // check and save output tensor

  // Returns
  return std::tuple<paddle::Tensor,
                    paddle::Tensor&,
                    paddle::Tensor&,
                    paddle::Tensor,
                    paddle::Tensor,
                    paddle::Tensor>{
      out, mean_out, variance_out, saved_mean, saved_variance, reserve_space};
}

namespace sparse {

std::tuple<paddle::Tensor,
           paddle::Tensor&,
           paddle::Tensor&,
           paddle::Tensor,
           paddle::Tensor,
           paddle::Tensor>
sync_batch_norm__ad_func(const paddle::Tensor& x,
                         const paddle::Tensor& scale,
                         const paddle::Tensor& bias,
                         paddle::Tensor& mean,      // NOLINT
                         paddle::Tensor& variance,  // NOLINT
                         float momentum,
                         float epsilon,
                         std::string data_layout,
                         bool is_test,
                         bool use_global_stats,
                         bool trainable_statistics,
                         bool fuse_with_relu) {
  VLOG(3) << "Running AD API: "
          << "sync_batch_norm_";
  // Dygraph Record Event
  paddle::platform::RecordEvent dygraph_entrance_record_event(
      "sync_batch_norm_ dygraph",
      paddle::platform::TracerEventType::Operator,
      1);

  // AMP Logic

  VLOG(5) << " No AMP for sync_batch_norm__ad_func because it is a inplace or "
             "cast api. ";
  // Layout autotune

  if (egr::Controller::Instance().UseLayoutAutoTune()) {
    paddle::small_vector<std::vector<paddle::Tensor>, egr::kSlotSmallVectorSize>
        tensors_vector = {{x}, {scale}, {bias}, {mean}, {variance}};

    auto op_name = phi::TransToFluidOpName("sync_batch_norm_");
    auto transformer = egr::EagerLayoutAutotune<std::string>(
        op_name, tensors_vector, &data_layout);
    auto new_x = transformer->TransInTensor("x", x);
    auto new_scale = transformer->TransInTensor("scale", scale);
    auto new_bias = transformer->TransInTensor("bias", bias);
    auto new_mean = transformer->TransInTensor("mean", mean);
    auto new_variance = transformer->TransInTensor("variance", variance);

    VLOG(5) << "Check and Prepare For LAYOUT " << op_name;
    paddle::imperative::LayoutAutotuneGuard guard(
        egr::Controller::Instance().GetCurrentTracer(), false);
    std::tuple<paddle::Tensor,
               paddle::Tensor&,
               paddle::Tensor&,
               paddle::Tensor,
               paddle::Tensor,
               paddle::Tensor>
        api_result = sync_batch_norm__ad_func(new_x,
                                              new_scale,
                                              new_bias,
                                              new_mean,
                                              new_variance,
                                              momentum,
                                              epsilon,
                                              data_layout,
                                              is_test,
                                              use_global_stats,
                                              trainable_statistics,
                                              fuse_with_relu);

    auto& out = std::get<0>(api_result);
    transformer->SetOutTensorLayout(&out);
    auto& mean_out = std::get<1>(api_result);
    transformer->SetOutTensorLayout(&mean_out);
    auto& variance_out = std::get<2>(api_result);
    transformer->SetOutTensorLayout(&variance_out);
    auto& saved_mean = std::get<3>(api_result);
    transformer->SetOutTensorLayout(&saved_mean);
    auto& saved_variance = std::get<4>(api_result);
    transformer->SetOutTensorLayout(&saved_variance);
    auto& reserve_space = std::get<5>(api_result);
    transformer->SetOutTensorLayout(&reserve_space);

    // Returns
    return std::tuple<paddle::Tensor,
                      paddle::Tensor&,
                      paddle::Tensor&,
                      paddle::Tensor,
                      paddle::Tensor,
                      paddle::Tensor>{
        out, mean_out, variance_out, saved_mean, saved_variance, reserve_space};
  }

  // Get Input AutoGradMeta
  egr::AutogradMeta* x_autograd_meta =
      egr::EagerUtils::nullable_autograd_meta(x);
  egr::AutogradMeta* scale_autograd_meta =
      egr::EagerUtils::nullable_autograd_meta(scale);
  egr::AutogradMeta* bias_autograd_meta =
      egr::EagerUtils::nullable_autograd_meta(bias);
  egr::AutogradMeta* mean_autograd_meta =
      egr::EagerUtils::nullable_autograd_meta(mean);
  egr::AutogradMeta* variance_autograd_meta =
      egr::EagerUtils::nullable_autograd_meta(variance);

  VLOG(5) << "Running C++ API: "
          << "sync_batch_norm_";
  // Before log info

  if (VLOG_IS_ON(3)) {
    const char* INPUT_PRINT_TEMPLATE = "{ Input: [%s]} ";

    std::string input_str = "";
    std::string output_str = "";
    const char* TENSOR_X_TEMPLATE = "(x, [%s]), ";
    std::string input_x_str = paddle::string::Sprintf(
        TENSOR_X_TEMPLATE, egr::EagerUtils::TensorStr(x));
    input_str += input_x_str;
    const char* TENSOR_SCALE_TEMPLATE = "(scale, [%s]), ";
    std::string input_scale_str = paddle::string::Sprintf(
        TENSOR_SCALE_TEMPLATE, egr::EagerUtils::TensorStr(scale));
    input_str += input_scale_str;
    const char* TENSOR_BIAS_TEMPLATE = "(bias, [%s]), ";
    std::string input_bias_str = paddle::string::Sprintf(
        TENSOR_BIAS_TEMPLATE, egr::EagerUtils::TensorStr(bias));
    input_str += input_bias_str;
    const char* TENSOR_MEAN_TEMPLATE = "(mean, [%s]), ";
    std::string input_mean_str = paddle::string::Sprintf(
        TENSOR_MEAN_TEMPLATE, egr::EagerUtils::TensorStr(mean));
    input_str += input_mean_str;
    const char* TENSOR_VARIANCE_TEMPLATE = "(variance, [%s]), ";
    std::string input_variance_str = paddle::string::Sprintf(
        TENSOR_VARIANCE_TEMPLATE, egr::EagerUtils::TensorStr(variance));
    input_str += input_variance_str;
    VLOG(3) << paddle::string::Sprintf(INPUT_PRINT_TEMPLATE, input_str);
  }

  // Forward API Call
  auto api_result =
      paddle::experimental::sparse::sync_batch_norm_(x,
                                                     scale,
                                                     bias,
                                                     mean,
                                                     variance,
                                                     momentum,
                                                     epsilon,
                                                     data_layout,
                                                     is_test,
                                                     use_global_stats,
                                                     trainable_statistics,
                                                     fuse_with_relu);
  // Check NaN and Inf if needed
  if (FLAGS_check_nan_inf) {
    egr::CheckTensorHasNanOrInf("sync_batch_norm_", api_result);
  }

  // Get Outputs
  auto& out = std::get<0>(api_result);
  auto& mean_out = std::get<1>(api_result);
  auto& variance_out = std::get<2>(api_result);
  auto& saved_mean = std::get<3>(api_result);
  auto& saved_variance = std::get<4>(api_result);
  auto& reserve_space = std::get<5>(api_result);

  // Get Output AutoGradMeta
  egr::AutogradMeta* out_autograd_meta = egr::EagerUtils::autograd_meta(&out);
  egr::AutogradMeta* mean_out_autograd_meta =
      egr::EagerUtils::autograd_meta(&mean_out);
  egr::AutogradMeta* variance_out_autograd_meta =
      egr::EagerUtils::autograd_meta(&variance_out);
  egr::AutogradMeta* saved_mean_autograd_meta =
      egr::EagerUtils::autograd_meta(&saved_mean);
  egr::AutogradMeta* saved_variance_autograd_meta =
      egr::EagerUtils::autograd_meta(&saved_variance);
  egr::AutogradMeta* reserve_space_autograd_meta =
      egr::EagerUtils::autograd_meta(&reserve_space);
  bool trace_backward = egr::Controller::Instance().HasGrad();
  bool require_any_grad =
      egr::EagerUtils::ComputeRequireGrad(trace_backward,
                                          x_autograd_meta,
                                          scale_autograd_meta,
                                          bias_autograd_meta,
                                          mean_autograd_meta,
                                          variance_autograd_meta);

  // Check Inplace if needed

  egr::EagerUtils::CheckInplace(mean, mean_autograd_meta, require_any_grad);

  egr::EagerUtils::CheckInplace(
      variance, variance_autograd_meta, require_any_grad);

  // Bump Inplace Version
  mean.bump_inplace_version();
  VLOG(3) << "Tensor(" << mean.name() << ") uses Inplace Strategy.";

  // Bump Inplace Version
  variance.bump_inplace_version();
  VLOG(3) << "Tensor(" << variance.name() << ") uses Inplace Strategy.";

  // Node Creation
  if (require_any_grad) {
    paddle::platform::RecordEvent node_creation_record_event(
        "sync_batch_norm_ node_creation",
        paddle::platform::TracerEventType::OperatorInner,
        1);

    egr::EagerUtils::PassStopGradient(false,
                                      out_autograd_meta,
                                      mean_out_autograd_meta,
                                      variance_out_autograd_meta,
                                      saved_mean_autograd_meta,
                                      saved_variance_autograd_meta,
                                      reserve_space_autograd_meta);

    // Node Construction
    auto grad_node =
        std::shared_ptr<SyncBatchNormGradNode>(new SyncBatchNormGradNode(6, 5));

    egr::Controller::Instance().PushBackForceSequentialNodes(grad_node.get());

    // SetAttributes if needed
    grad_node->SetAttributemomentum(momentum);
    grad_node->SetAttributeepsilon(epsilon);
    grad_node->SetAttributedata_layout(data_layout);
    grad_node->SetAttributeis_test(is_test);
    grad_node->SetAttributeuse_global_stats(use_global_stats);
    grad_node->SetAttributetrainable_statistics(trainable_statistics);
    grad_node->SetAttributefuse_with_relu(fuse_with_relu);
    // Set TensorWrappers for Forward Inputs if needed
    grad_node->SetTensorWrapperx(x);
    grad_node->SetTensorWrapperscale(scale);
    grad_node->SetTensorWrapperbias(bias);
    // SetGradOutMeta & SetEdges
    grad_node->SetGradOutMeta(x, 0);
    grad_node->SetGradOutMeta(scale, 1);
    grad_node->SetGradOutMeta(bias, 2);
    // SetOutRank & SetHistory & SetGradInMeta & RetainGrad
    if (out_autograd_meta) {
      egr::EagerUtils::SetOutRankWithSlot(out_autograd_meta, 0);
    }
    if (mean_out_autograd_meta) {
      egr::EagerUtils::SetOutRankWithSlot(mean_out_autograd_meta, 1);
    }
    if (variance_out_autograd_meta) {
      egr::EagerUtils::SetOutRankWithSlot(variance_out_autograd_meta, 2);
    }
    if (saved_mean_autograd_meta) {
      egr::EagerUtils::SetOutRankWithSlot(saved_mean_autograd_meta, 3);
    }
    if (saved_variance_autograd_meta) {
      egr::EagerUtils::SetOutRankWithSlot(saved_variance_autograd_meta, 4);
    }
    if (reserve_space_autograd_meta) {
      egr::EagerUtils::SetOutRankWithSlot(reserve_space_autograd_meta, 5);
    }
    if (out_autograd_meta) {
      egr::EagerUtils::SetHistory(out_autograd_meta, grad_node);
    }
    if (mean_out_autograd_meta) {
      egr::EagerUtils::SetHistory(mean_out_autograd_meta, grad_node);
    }
    if (variance_out_autograd_meta) {
      egr::EagerUtils::SetHistory(variance_out_autograd_meta, grad_node);
    }
    if (saved_mean_autograd_meta) {
      egr::EagerUtils::SetHistory(saved_mean_autograd_meta, grad_node);
    }
    if (saved_variance_autograd_meta) {
      egr::EagerUtils::SetHistory(saved_variance_autograd_meta, grad_node);
    }
    if (reserve_space_autograd_meta) {
      egr::EagerUtils::SetHistory(reserve_space_autograd_meta, grad_node);
    }
    grad_node->SetGradInMeta(out, 0);
    grad_node->SetGradInMeta(mean_out, 1);
    grad_node->SetGradInMeta(variance_out, 2);
    grad_node->SetGradInMeta(saved_mean, 3);
    grad_node->SetGradInMeta(saved_variance, 4);
    grad_node->SetGradInMeta(reserve_space, 5);
    egr::EagerUtils::CheckAndRetainGrad(out);
    egr::EagerUtils::CheckAndRetainGrad(mean_out);
    egr::EagerUtils::CheckAndRetainGrad(variance_out);
    egr::EagerUtils::CheckAndRetainGrad(saved_mean);
    egr::EagerUtils::CheckAndRetainGrad(saved_variance);
    egr::EagerUtils::CheckAndRetainGrad(reserve_space);
    // Set TensorWrappers for Forward Outputs if needed
    grad_node->SetTensorWrappersaved_mean(saved_mean);
    grad_node->SetTensorWrappersaved_variance(saved_variance);
    grad_node->SetTensorWrapperreserve_space(reserve_space);
  }

  VLOG(4) << "Finish AD API: sync_batch_norm_";
  // LOG IF DEBUG

  if (VLOG_IS_ON(4)) {
    const char* INPUT_PRINT_TEMPLATE = "{ Input: [%s],  Output: [%s] } ";

    std::string input_str = "";
    std::string output_str = "";
    const char* TENSOR_X_TEMPLATE = "(x, [%s]), ";
    std::string input_x_str = paddle::string::Sprintf(
        TENSOR_X_TEMPLATE, egr::EagerUtils::TensorStr(x));
    input_str += input_x_str;
    const char* TENSOR_SCALE_TEMPLATE = "(scale, [%s]), ";
    std::string input_scale_str = paddle::string::Sprintf(
        TENSOR_SCALE_TEMPLATE, egr::EagerUtils::TensorStr(scale));
    input_str += input_scale_str;
    const char* TENSOR_BIAS_TEMPLATE = "(bias, [%s]), ";
    std::string input_bias_str = paddle::string::Sprintf(
        TENSOR_BIAS_TEMPLATE, egr::EagerUtils::TensorStr(bias));
    input_str += input_bias_str;
    const char* TENSOR_MEAN_TEMPLATE = "(mean, [%s]), ";
    std::string input_mean_str = paddle::string::Sprintf(
        TENSOR_MEAN_TEMPLATE, egr::EagerUtils::TensorStr(mean));
    input_str += input_mean_str;
    const char* TENSOR_VARIANCE_TEMPLATE = "(variance, [%s]), ";
    std::string input_variance_str = paddle::string::Sprintf(
        TENSOR_VARIANCE_TEMPLATE, egr::EagerUtils::TensorStr(variance));
    input_str += input_variance_str;
    const char* TENSOR_OUT_TEMPLATE = "(out, [%s]), ";
    std::string output_out_str = paddle::string::Sprintf(
        TENSOR_OUT_TEMPLATE, egr::EagerUtils::TensorStr(out));
    output_str += output_out_str;
    const char* TENSOR_MEAN_OUT_TEMPLATE = "(mean_out, [%s]), ";
    std::string output_mean_out_str = paddle::string::Sprintf(
        TENSOR_MEAN_OUT_TEMPLATE, egr::EagerUtils::TensorStr(mean_out));
    output_str += output_mean_out_str;
    const char* TENSOR_VARIANCE_OUT_TEMPLATE = "(variance_out, [%s]), ";
    std::string output_variance_out_str = paddle::string::Sprintf(
        TENSOR_VARIANCE_OUT_TEMPLATE, egr::EagerUtils::TensorStr(variance_out));
    output_str += output_variance_out_str;
    const char* TENSOR_SAVED_MEAN_TEMPLATE = "(saved_mean, [%s]), ";
    std::string output_saved_mean_str = paddle::string::Sprintf(
        TENSOR_SAVED_MEAN_TEMPLATE, egr::EagerUtils::TensorStr(saved_mean));
    output_str += output_saved_mean_str;
    const char* TENSOR_SAVED_VARIANCE_TEMPLATE = "(saved_variance, [%s]), ";
    std::string output_saved_variance_str =
        paddle::string::Sprintf(TENSOR_SAVED_VARIANCE_TEMPLATE,
                                egr::EagerUtils::TensorStr(saved_variance));
    output_str += output_saved_variance_str;
    const char* TENSOR_RESERVE_SPACE_TEMPLATE = "(reserve_space, [%s]), ";
    std::string output_reserve_space_str =
        paddle::string::Sprintf(TENSOR_RESERVE_SPACE_TEMPLATE,
                                egr::EagerUtils::TensorStr(reserve_space));
    output_str += output_reserve_space_str;
    VLOG(4) << paddle::string::Sprintf(
        INPUT_PRINT_TEMPLATE, input_str, output_str);
  }

  // Returns
  return std::tuple<paddle::Tensor,
                    paddle::Tensor&,
                    paddle::Tensor&,
                    paddle::Tensor,
                    paddle::Tensor,
                    paddle::Tensor>{
      out, mean_out, variance_out, saved_mean, saved_variance, reserve_space};
}

}  // namespace sparse
