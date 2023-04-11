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
#include "paddle/fluid/platform/profiler/event_tracing.h"

DECLARE_bool(check_nan_inf);

paddle::Tensor conv2d_ad_func(const paddle::Tensor& input,
                              const paddle::Tensor& filter,
                              std::vector<int> strides,
                              std::vector<int> paddings,
                              std::string padding_algorithm,
                              std::vector<int> dilations,
                              int groups,
                              std::string data_format) {
  // Dygraph Record Event
  paddle::platform::RecordEvent dygraph_entrance_record_event(
      "conv2d dygraph", paddle::platform::TracerEventType::Operator, 1);

  // AMP Logic
  if (egr::Controller::Instance().GetAMPLevel() !=
      paddle::imperative::AmpLevel::O0) {
    VLOG(5) << "Check and Prepare For AMP";
    auto op_name = phi::TransToFluidOpName("conv2d");
    paddle::small_vector<std::vector<paddle::Tensor>, egr::kSlotSmallVectorSize>
        amp_tensors_vector = {{input}, {filter}};

    auto amp_dst_dtype = egr::GetAmpDestDtype(op_name, amp_tensors_vector);

    auto new_input =
        egr::EagerAmpAutoCast("input", input, amp_dst_dtype, op_name);
    auto new_filter =
        egr::EagerAmpAutoCast("filter", filter, amp_dst_dtype, op_name);

    {
      paddle::imperative::AutoCastGuard guard(
          egr::Controller::Instance().GetCurrentTracer(),
          paddle::imperative::AmpLevel::O0);
      return conv2d_ad_func(new_input,
                            new_filter,
                            strides,
                            paddings,
                            padding_algorithm,
                            dilations,
                            groups,
                            data_format);
    }
  }

  // Layout autotune

  if (egr::Controller::Instance().UseLayoutAutoTune()) {
    VLOG(5) << "Check and Prepare For LAYOUT";
    paddle::small_vector<std::vector<paddle::Tensor>, egr::kSlotSmallVectorSize>
        tensors_vector = {{input}, {filter}};

    auto op_name = phi::TransToFluidOpName("conv2d");
    auto transformer = egr::EagerLayoutAutotune<std::string>(
        op_name, tensors_vector, &data_format);
    auto new_input = transformer->TransInTensor("input", input);
    bool need_tune = egr::Controller::Instance().UseLayoutAutoTune();
    egr::Controller::Instance().DisableLayoutAutoTune();
    auto out = conv2d_ad_func(new_input,
                              filter,
                              strides,
                              paddings,
                              padding_algorithm,
                              dilations,
                              groups,
                              data_format);
    transformer->SetOutTensorLayout(&out);
    if (need_tune) {
      egr::Controller::Instance().EnableLayoutAutoTune();
    }
    // Returns
    return out;
  }

  // Get Input AutoGradMeta
  egr::AutogradMeta* input_autograd_meta =
      egr::EagerUtils::nullable_autograd_meta(input);
  egr::AutogradMeta* filter_autograd_meta =
      egr::EagerUtils::nullable_autograd_meta(filter);
  // Forward API Call
  VLOG(3) << "Final State Running: "
          << "conv2d_ad_func";
  auto api_result = paddle::experimental::conv2d(input,
                                                 filter,
                                                 strides,
                                                 paddings,
                                                 padding_algorithm,
                                                 dilations,
                                                 groups,
                                                 data_format);
  std::string forward_trace = "";
  // Check NaN and Inf if needed
  if (FLAGS_check_nan_inf) {
    egr::CheckTensorHasNanOrInf("conv2d", api_result);
    std::string filename = __FILE__;
    std::string line = std::to_string(__LINE__);
    std::string function_name = __FUNCTION__;
    forward_trace = filename + " " + line + " " + function_name + "\n";
    forward_trace =
        egr::Controller::Instance().GetOpPythonStackStr() + forward_trace;
    try {
      PADDLE_ENFORCE(false,
                     "conv2d's backward has nan/inf, please check the data of "
                     "backward op");
    } catch (std::exception& e) {
      egr::Controller::Instance().SetOpPythonStackStr(forward_trace);
    }
  }

  // Get Outputs
  auto& out = api_result;

  // Get Output AutoGradMeta
  egr::AutogradMeta* out_autograd_meta = egr::EagerUtils::autograd_meta(&out);
  bool trace_backward = egr::Controller::Instance().HasGrad();
  bool require_any_grad = egr::EagerUtils::ComputeRequireGrad(
      trace_backward, input_autograd_meta, filter_autograd_meta);

  // Check Inplace if needed

  // Node Creation
  if (require_any_grad) {
    paddle::platform::RecordEvent node_creation_record_event(
        "conv2d node_creation",
        paddle::platform::TracerEventType::OperatorInner,
        1);

    egr::EagerUtils::PassStopGradient(false, out_autograd_meta);

    // Node Construction
    auto grad_node =
        std::shared_ptr<Conv2dGradNodeFinal>(new Conv2dGradNodeFinal(1, 2));

    // Set forward's stack
    if (FLAGS_check_nan_inf) {
      grad_node->SetForwardTrace(forward_trace);
    }

    // SetAttributes if needed
    grad_node->SetAttributestrides(strides);
    grad_node->SetAttributepaddings(paddings);
    grad_node->SetAttributepadding_algorithm(padding_algorithm);
    grad_node->SetAttributegroups(groups);
    grad_node->SetAttributedilations(dilations);
    grad_node->SetAttributedata_format(data_format);
    // Set TensorWrappers for Forward Inputs if needed
    grad_node->SetTensorWrapperinput(input);
    grad_node->SetTensorWrapperfilter(filter);
    // SetGradOutMeta & SetEdges
    grad_node->SetGradOutMeta(input, 0);
    grad_node->SetGradOutMeta(filter, 1);
    // SetOutRank & SetHistory & SetGradInMeta & RetainGrad
    if (out_autograd_meta) {
      egr::EagerUtils::SetOutRankWithSlot(out_autograd_meta, 0);
    }
    if (out_autograd_meta) {
      egr::EagerUtils::SetHistory(out_autograd_meta, grad_node);
    }
    grad_node->SetGradInMeta(out, 0);
    // Set TensorWrappers for Forward Outputs if needed
  }

  // Returns
  return out;
}
