// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

// Pre-Processing function.
// The function here will be called by the functions in
// paddle/fluid/pybind/static_op_function.cc and
// paddle/fluid/pybind/eager_op_function.cc. Mainly used to customize the
// processing of parameters originally done in the Python API
#include "paddle/fluid/pybind/arg_pre_process.h"
#include "paddle/common/ddim.h"
#include "paddle/fluid/eager/utils.h"
#include "paddle/fluid/pir/utils/general_functions.h"
#include "paddle/fluid/pybind/eager_utils.h"
#include "paddle/fluid/pybind/op_function_common.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/enforce.h"

namespace paddle {
namespace pybind {
constexpr char kStopGradientAttrName[] = "stop_gradient";  // NOLINT
void ExpandAsPreProcess(paddle::Tensor* x,
                        paddle::optional<paddle::Tensor>* y,
                        std::vector<int64_t>* target_shape) {
  if (target_shape->empty() && y->get_ptr() == nullptr) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "The y of expand_as api must be specified."));
  }
  if (y->get_ptr() == nullptr) return;
  *target_shape = common::vectorize<int64_t>(y->get_ptr()->dims());
}
void ExpandAsPreProcess(pir::Value* x,
                        paddle::optional<pir::Value>* y,
                        std::vector<int64_t>* target_shape) {
  if (target_shape->empty() && y->get_ptr() == nullptr) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "The y of expand_as api must be specified."));
  }
  if (y->get_ptr() == nullptr) return;
  *target_shape = pir::GetShapeFromValue(*(y->get_ptr()));

  /**
   * if convert_dtype(x.dtype) == 'bool' and not x.stop_gradient:
   *    raise ValueError(
   *        "When the data type of input 'x' for expand_as is bool, "
   *        "you must set its stop_gradient to be False by "
   *        "some_var.stop_gradient = True, supporting "
   *        "some_var as the input 'x'."
   *    )
   *
   */
  auto dtype = pir::GetValueDtype(*x);
  auto stop_gradient_attr =
      x->attribute<pir::BoolAttribute>(kStopGradientAttrName);
  auto stop_gradient = !stop_gradient_attr || stop_gradient_attr.data();
  if (dtype == phi::DataType::BOOL && !stop_gradient) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "When the data type of input 'x' for expand_as is bool, "
        "you must set its stop_gradient to be False by "
        "some_var.stop_gradient = True, supporting "
        "some_var as the input 'x'."));
  }
}

}  // namespace pybind

}  // namespace paddle
