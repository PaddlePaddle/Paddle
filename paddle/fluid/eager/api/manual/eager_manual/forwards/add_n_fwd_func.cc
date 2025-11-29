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
#include "paddle/fluid/eager/nan_inf_utils.h"
#include "paddle/fluid/imperative/amp_utils.h"
#include "paddle/phi/core/platform/profiler/event_tracing.h"

COMMON_DECLARE_bool(check_nan_inf);
COMMON_DECLARE_bool(check_cuda_error);
COMMON_DECLARE_bool(enable_unique_name);
COMMON_DECLARE_string(tensor_md5_checksum_output_path);
COMMON_DECLARE_string(dump_api_python_stack_path);

#define SEPARATOR "=========================="
paddle::Tensor add_n_ad_func(const std::vector<paddle::Tensor>& x,
                             paddle::optional<paddle::Tensor*> predefined_out) {
  VLOG(3) << "\n"
          << SEPARATOR << "Running_AD_API: "
          << "add_n" << SEPARATOR;
  if (FLAGS_check_cuda_error) [[unlikely]] {
    egr::CUDAErrorCheck("add_n_ad_func begin");
  }
  // Dygraph Record Event
  phi::RecordEvent dygraph_entrance_record_event(
      "add_n dygraph", phi::TracerEventType::Operator, 1);

  // AMP Logic
  if (egr::Controller::Instance().GetAMPLevel() !=
      paddle::imperative::AmpLevel::O0) {
    VLOG(5) << "Check and Prepare For AMP, AMP Level : "
            << static_cast<int>(egr::Controller::Instance().GetAMPLevel());
    auto op_name = phi::TransToFluidOpName("add_n");
    paddle::small_vector<std::vector<paddle::Tensor>, egr::kSlotSmallVectorSize>
        amp_tensors_vector = {x};

    auto amp_dst_dtype =
        paddle::imperative::GetAmpDestDtype(op_name, amp_tensors_vector);
    VLOG(5) << "AMP Get Dest Dtype : " << amp_dst_dtype;
    auto NEW_x =
        paddle::imperative::AmpAutoCasts("x", x, amp_dst_dtype, op_name);

    {
      paddle::imperative::AutoCastGuard guard(
          egr::Controller::Instance().GetCurrentAmpAttrs(),
          paddle::imperative::AmpLevel::O0);
      return add_n_ad_func(NEW_x);
    }
  }

  // Get Input AutoGradMeta
  std::vector<egr::AutogradMeta*> x_autograd_meta_vec =
      egr::EagerUtils::nullable_autograd_meta(x);
  std::vector<egr::AutogradMeta*>* x_autograd_meta = &x_autograd_meta_vec;
  // Check LeafTensor if its GradNodeAccumulation TensorMeta is consistent with
  // its TensorMeta
  egr::CheckGradNodeAccumulation(x);
  // Forward API Call
  std::string unique_api_name;
  if (VLOG_IS_ON(3) || FLAGS_enable_unique_name) {
    static int64_t call_count = 0;
    call_count++;
    unique_api_name = egr::GenerateUniqueApiName("add_n", call_count);
  }
  // Save forward call stack to file for debug
  if (FLAGS_call_stack_level == 3 &&
      !FLAGS_dump_api_python_stack_path.empty()) {
    egr::SavePythonCallStackToFile(FLAGS_dump_api_python_stack_path,
                                   unique_api_name);
  }
  VLOG(3) << "\n"
          << SEPARATOR << "Running_C++_API: " << unique_api_name << SEPARATOR;
  auto api_result = paddle::experimental::add_n(x);
  VLOG(3) << "\n"
          << SEPARATOR << "Finish_C++_API: " << unique_api_name << SEPARATOR;
  // Check NaN and Inf if needed
  if (FLAGS_check_nan_inf) {
    egr::CheckTensorHasNanOrInf("add_n", api_result);
  }

  // Get Outputs
  auto& out = api_result;
  if (VLOG_IS_ON(6) || FLAGS_enable_unique_name) {
    egr::SetTensorName(unique_api_name, "out", &out);
  }
  if (!FLAGS_tensor_md5_checksum_output_path.empty()) {
    egr::SaveTensorMD5CheckSumToFile(FLAGS_tensor_md5_checksum_output_path,
                                     out);
  }
  // Get Output AutoGradMeta
  egr::AutogradMeta* out_autograd_meta = egr::EagerUtils::autograd_meta(&out);
  bool trace_backward = egr::Controller::Instance().HasGrad();
  bool require_any_grad =
      egr::EagerUtils::ComputeRequireGrad(trace_backward, x_autograd_meta);

  // Check Inplace if needed

  // Node Creation
  if (require_any_grad) {
    phi::RecordEvent node_creation_record_event(
        "add_n node_creation", phi::TracerEventType::OperatorInner, 1);

    egr::EagerUtils::PassStopGradient(false, out_autograd_meta);

    // Node Construction
    auto grad_node = std::shared_ptr<AddNGradNodeFinal>(  // NOLINT
        new AddNGradNodeFinal(1, 1));
    if (VLOG_IS_ON(6) || FLAGS_enable_unique_name) {
      // Set GradNodeName
      grad_node->SetNameFromAPI(unique_api_name);
    }
    // Set forward's stack
    if (FLAGS_check_nan_inf) {
      grad_node->SetForwardTrace(egr::Controller::Instance().GetPythonStack());
    }
    // Set for Record Subgraph
    if (egr::EagerBackwardSubGraphNodeRecorder::Instance()
            .NeedCaptureSubGraph()) {
      VLOG(3) << "Capture the grad node" << grad_node->name() << "("
              << grad_node.get() << ")"
              << "for subgraph.";
      egr::EagerBackwardSubGraphNodeRecorder::Instance().AddGradNode(
          grad_node.get());
    }

    // SetAttributes if needed

    // Set TensorWrappers for Forward Inputs if needed
    grad_node->SetTensorWrapper_x(x);
    // SetGradOutMeta & SetEdges
    grad_node->SetGradOutMeta(x, 0);
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
  if (VLOG_IS_ON(3)) {
    const char* INPUT_PRINT_TEMPLATE =
        "\nForward Debug Info {\nAPI_Name: %s \nInput: [%s]  \nOutput: [%s] } ";

    std::string input_str = "";
    std::string output_str = "";
    const char* TENSOR_X_TEMPLATE = " \n( x , %s), ";
    std::string input_x_str = paddle::string::Sprintf(
        TENSOR_X_TEMPLATE, egr::EagerUtils::TensorStr(x));
    input_str += input_x_str;
    const char* TENSOR_OUT_TEMPLATE = " \n( out , %s), ";
    std::string output_out_str = paddle::string::Sprintf(
        TENSOR_OUT_TEMPLATE, egr::EagerUtils::TensorStr(out));
    output_str += output_out_str;
    VLOG(3) << paddle::string::Sprintf(
        INPUT_PRINT_TEMPLATE, unique_api_name, input_str, output_str);
  }

  if (FLAGS_check_cuda_error) [[unlikely]] {
    egr::CUDAErrorCheck("add_n_ad_func finish");
  }
  VLOG(3) << "\n"
          << SEPARATOR << "Finish_AD_API: "
          << "add_n" << SEPARATOR;
  // Returns
  return out;
}
