// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/eager/api/manual/eager_manual/dygraph_forward_api.h"
#include "paddle/fluid/eager/api/manual/eager_manual/nodes/nodes.h"
#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/eager/utils.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/phi/core/platform/profiler/event_tracing.h"

paddle::small_vector<std::vector<paddle::Tensor>,
                     egr::kSlotSmallVectorSize>  // NOLINT
DtensorToLocalGradNode::operator()(
    paddle::small_vector<std::vector<paddle::Tensor>,
                         egr::kSlotSmallVectorSize>& grads,
    bool create_graph,
    bool is_new_grad) {
#ifdef PADDLE_WITH_DISTRIBUTE
  VLOG(3) << "Running AD API GRAD: "
          << "dtensor_to_local";

  // This 'Local_XXXGradNode' record event is different with
  // 'Global_XXXGradNode' event.
  // * 'Local_XXXGradNode' will only cover execution time of this function.
  // * 'Global_XXXGradNode' will not only cover execution time of this function,
  // but also include gradient
  //    accumulation when the output(s) of corresponding forward OP are shared
  //    by other OP(s), which may have extra accumulation overhead than
  //    'Local_XXXGradNode'.
  phi::RecordEvent node_execution_inner(
      "Local_DtensorToLocalGradNode", phi::TracerEventType::OperatorInner, 1);

  // Apply Gradient Hooks
  auto hooked_grad = ApplyGradientHooks(grads);

  // Collect GradIn Tensors, Attrs and Recovered TensorWrappers
  auto input = egr::EagerUtils::RecoverTensorWrapper(&this->input_);
  const auto& dist_attr =
      std::static_pointer_cast<phi::distributed::DistTensor>(input.impl())
          ->dist_attr();
  auto& grad_out = hooked_grad[0][0];
  // Prepare Grad function call

  const auto& out_metas = OutputMeta();
  paddle::small_vector<std::vector<paddle::Tensor>, egr::kSlotSmallVectorSize>
      returns(1);

  out_metas[0].size() == 0 ? returns[0].resize(1)
                           : returns[0].resize(out_metas[0].size());

  auto& grad_input = returns[0][0];

  VLOG(5) << "Running C++ API: "
          << "dtensor_to_local_func";

  if (VLOG_IS_ON(3)) {
    const char* INPUT_PRINT_TEMPLATE = "{ Input: [%s]} ";

    std::string input_str = "";
    const char* TENSOR_OUT_GRAD_TEMPLATE = " \n( out_grad , [%s]), ";
    std::string input_out_grad_str = paddle::string::Sprintf(
        TENSOR_OUT_GRAD_TEMPLATE, egr::EagerUtils::TensorStr(grad_out));
    input_str += input_out_grad_str;
    const char* TENSOR_X_TEMPLATE = " \n( x , [%s]), ";
    std::string input_x_str = paddle::string::Sprintf(
        TENSOR_X_TEMPLATE, egr::EagerUtils::TensorStr(input));
    input_str += input_x_str;
    VLOG(3) << paddle::string::Sprintf(INPUT_PRINT_TEMPLATE, input_str);
  }

  // Backward call dtensor_to_local_func function
  auto dist_grad_ptr = std::make_shared<phi::distributed::DistTensor>(
      grad_out.dims(), dist_attr);

  *(dist_grad_ptr->unsafe_mutable_value()) =
      *(static_cast<phi::DenseTensor*>(grad_out.impl().get()));
  grad_input.set_impl(dist_grad_ptr);

  VLOG(5) << "Finish C++ API: dtensor_to_local_func";
  VLOG(6) << "gradnode_ptr = " << this;

  if (VLOG_IS_ON(4)) {
    const char* INPUT_PRINT_TEMPLATE = "{ Input: [%s],  \n Output: [%s] } ";
    std::string input_str = "";
    std::string output_str = "";
    const char* TENSOR_OUT_GRAD_TEMPLATE = " \n( out_grad , [%s]), ";
    std::string input_out_grad_str = paddle::string::Sprintf(
        TENSOR_OUT_GRAD_TEMPLATE, egr::EagerUtils::TensorStr(grad_out));
    input_str += input_out_grad_str;
    const char* TENSOR_X_TEMPLATE = " \n( x , [%s]), ";
    std::string input_x_str = paddle::string::Sprintf(
        TENSOR_X_TEMPLATE, egr::EagerUtils::TensorStr(input));
    input_str += input_x_str;
    const char* TENSOR_X_GRAD_TEMPLATE = " \n ( input_grad , [%s]), ";
    std::string output_x_grad_str = paddle::string::Sprintf(
        TENSOR_X_GRAD_TEMPLATE, egr::EagerUtils::TensorStr(grad_input));
    output_str += output_x_grad_str;
    VLOG(4) << paddle::string::Sprintf(
        INPUT_PRINT_TEMPLATE, input_str, output_str);
  }

  return returns;
#else
  PADDLE_THROW(common::errors::Unavailable(
      "DtensorToLocalGrad is not supported in this version of Paddle. Try to "
      "recompile it with WITH_DISTRIBUTE=ON and reinstall this package."));
  return paddle::small_vector<std::vector<paddle::Tensor>,
                              egr::kSlotSmallVectorSize>(1);
#endif
}
