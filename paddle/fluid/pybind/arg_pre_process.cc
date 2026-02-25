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
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/fluid/pir/utils/general_functions.h"
#include "paddle/fluid/pybind/eager_utils.h"
#include "paddle/fluid/pybind/op_function_common.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/kernels/funcs/common_infer_shape_functions.h"

namespace paddle {
namespace pybind {
constexpr char kStopGradientAttrName[] = "stop_gradient";  // NOLINT

// Helper to validate dimension equality for broadcast
static void ValidateBroadcastDim(int64_t actual,
                                 int64_t expected,
                                 const std::string& error_msg) {
  // In static graph, unknown dimensions are often represented as -1.
  if (actual < 0 || expected < 0) {
    return;
  }
  PADDLE_ENFORCE_EQ(actual == expected || actual == 1,
                    true,
                    phi::errors::InvalidArgument(
                        "%s But received actual = %ld, expected = %ld.",
                        error_msg,
                        actual,
                        expected));
}

static void CheckDataType(const std::string& op_name,
                          const std::string var_name,
                          const DataType& var_dtype,
                          const std::vector<DataType>& expect_dtype) {
  for (auto& t : expect_dtype) {
    if (var_dtype == t) return;
  }
  PADDLE_THROW(common::errors::InvalidType(
      "The dtype of %s of %s must be one of %s, but received %s.",
      var_name,
      op_name,
      phi::DataTypeToString(expect_dtype),
      phi::DataTypeToString(var_dtype)));
}
void ExpandAsPreProcess(Tensor* x,
                        paddle::optional<Tensor>* y,
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
  if (dtype == DataType::BOOL && !stop_gradient) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "When the data type of input 'x' for expand_as is bool, "
        "you must set its stop_gradient to be False by "
        "some_var.stop_gradient = True, supporting "
        "some_var as the input 'x'."));
  }
}
void RollPreProcess(Tensor* x, IntArray* shifts, IntVector* axis) {
  int64_t len_origin_shape = x->dims().size();
  if (axis != NULL) {
    int64_t axis_len = axis->size();
    for (int64_t i = 0; i < axis_len; i++) {
      PADDLE_ENFORCE_EQ(
          ((*axis)[i] < len_origin_shape && (*axis)[i] >= -len_origin_shape),
          true,
          common::errors::InvalidArgument("axis is out of range, it should be "
                                          "in range [%d, %d), but received %ld",
                                          -len_origin_shape,
                                          len_origin_shape,
                                          (*axis)[i]));
    }
  } else {
    axis = new IntVector();
  }
}
void RollPreProcess(Value* x, Value* shifts, IntVector* axis) {
  std::vector<int64_t> x_shape = pir::GetShapeFromValue(*x);
  int64_t len_origin_shape = x_shape.size();
  if (axis != NULL) {
    int64_t axis_len = axis->size();
    for (int64_t i = 0; i < axis_len; i++) {
      PADDLE_ENFORCE_EQ(
          ((*axis)[i] < len_origin_shape && (*axis)[i] >= -len_origin_shape),
          true,
          common::errors::InvalidArgument("axis is out of range, it should be "
                                          "in range [%d, %d), but received %ld",
                                          -len_origin_shape,
                                          len_origin_shape,
                                          (*axis)[i]));
    }
  } else {
    axis = new IntVector();
  }
}

void LogsumexpPreProcess(Tensor* x, std::vector<int>* axis, bool* reduce_all) {
  /**
  if axis == [] or len(axis) == len(x.shape):
      reduce_all = True
  else:
      reduce_all = False
  */
  if (axis->empty() || axis->size() == x->dims().size()) {
    *reduce_all = true;
  } else {
    *reduce_all = false;
  }
  return;
}

void LogsumexpPreProcess(pir::Value* x,
                         std::vector<int>* axis,
                         bool* reduce_all) {
  std::vector<int64_t> x_shape = pir::GetShapeFromValue(*x);
  if (axis->empty() || axis->size() == x_shape.size()) {
    *reduce_all = true;
  } else {
    *reduce_all = false;
  }
  return;
}
void SumPreProcess(Value* x, Value* axis) {
  paddle::dialect::SetStopGradient(axis);
}

void BinCountPreProcess(Tensor* x,
                        paddle::optional<Tensor>* weights,
                        Scalar* minlength) {
  CheckDataType(
      "bincount", "x", x->dtype(), {DataType::INT32, DataType::INT64});
}

void BinCountPreProcess(Value* x,
                        paddle::optional<Value>* weights,
                        Value* minlength) {
  CheckDataType("bincount",
                "x",
                pir::GetValueDtype(*x),
                {DataType::INT32, DataType::INT64});
}

void IsClosePreProcess(Value* x, Value* y, Value* rtol, Value* atol) {
  /*
  if in_pir_mode():
     check_variable_and_dtype(
         x,
         "input",
         ['float16', 'float32', 'float64', 'complex64', 'complex128'],
         'isclose',
     )
     check_variable_and_dtype(
         y,
         "input",
         ['float16', 'float32', 'float64', 'complex64', 'complex128'],
         'isclose',
     )
     if isinstance(rtol, paddle.pir.Value):
         check_variable_and_dtype(
             rtol,
             "input",
             ['float64'],
             'isclose',
         )
     else:
         check_type(rtol, 'rtol', float, 'isclose')
     if isinstance(atol, paddle.pir.Value):
         check_variable_and_dtype(
             atol,
             "input",
             ['float64'],
             'isclose',
         )
     else:
         check_type(atol, 'atol', float, 'isclose')

  */
  // 'float16', 'float32', 'float64', 'complex64', 'complex128'
  CheckDataType("is_close",
                "x",
                pir::GetValueDtype(*x),
                {DataType::FLOAT16,
                 DataType::FLOAT32,
                 DataType::FLOAT64,
                 DataType::COMPLEX64,
                 DataType::COMPLEX128});
  CheckDataType("is_close",
                "y",
                pir::GetValueDtype(*y),
                {DataType::FLOAT16,
                 DataType::FLOAT32,
                 DataType::FLOAT64,
                 DataType::COMPLEX64,
                 DataType::COMPLEX128});
  // 'float64'
  CheckDataType(
      "is_close", "rtol", pir::GetValueDtype(*rtol), {DataType::FLOAT64});
  CheckDataType(
      "is_close", "atol", pir::GetValueDtype(*atol), {DataType::FLOAT64});
}

void AllClosePreProcess(Value* x, Value* y, Value* rtol, Value* atol) {
  CheckDataType("allclose",
                "x",
                pir::GetValueDtype(*x),
                {DataType::BOOL,
                 DataType::INT32,
                 DataType::INT64,
                 DataType::FLOAT16,
                 DataType::FLOAT32,
                 DataType::FLOAT64});
  CheckDataType("allclose",
                "y",
                pir::GetValueDtype(*y),
                {DataType::BOOL,
                 DataType::INT32,
                 DataType::INT64,
                 DataType::FLOAT16,
                 DataType::FLOAT32,
                 DataType::FLOAT64});
  CheckDataType(
      "allclose", "rtol", pir::GetValueDtype(*rtol), {DataType::FLOAT64});
  CheckDataType(
      "allclose", "atol", pir::GetValueDtype(*atol), {DataType::FLOAT64});
}

void GridSamplePreProcess(Tensor* x,
                          Tensor* grid,
                          std::string* mode,
                          std::string* padding_mode,
                          bool* align_corners) {
  // mode should be in ['bilinear', 'nearest']
  // padding_mode should be in ['zeros', 'reflection', 'border']
  if (mode->compare("bilinear") != 0 && mode->compare("nearest") != 0) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "The mode of grid sample function should be in ['bilinear', "
        "'nearest'], but got: %s",
        *mode));
  }
  if (padding_mode->compare("zeros") != 0 &&
      padding_mode->compare("reflection") != 0 &&
      padding_mode->compare("border") != 0) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "The padding mode of grid sample function should be in "
        "['zeros', 'reflection', 'border'], but got: %s",
        *padding_mode));
  }
  return;
}

void GridSamplePreProcess(pir::Value* x,
                          pir::Value* grid,
                          std::string* mode,
                          std::string* padding_mode,
                          bool* align_corners) {
  // mode should be in ['bilinear', 'nearest']
  // padding_mode should be in ['zeros', 'reflection', 'border']
  if (mode->compare("bilinear") != 0 && mode->compare("nearest") != 0) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "The mode of grid sample function should be in ['bilinear', "
        "'nearest'], but got: %s",
        *mode));
  }
  if (padding_mode->compare("zeros") != 0 &&
      padding_mode->compare("reflection") != 0 &&
      padding_mode->compare("border") != 0) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "The padding mode of grid sample function should be in "
        "['zeros', 'reflection', 'border'], but got: %s",
        *padding_mode));
  }
  return;
}

// Addmm broadcast validation for dygraph
void AddmmPreProcess(Tensor* input, Tensor* x, Tensor* y) {
  auto input_shape = input->dims();
  auto x_shape = x->dims();
  auto y_shape = y->dims();

  // Validate x and y are 2D
  PADDLE_ENFORCE_EQ(
      x_shape.size(),
      2,
      phi::errors::InvalidArgument(
          "The dimension of x should be 2 but received x's shape: [%s]",
          x_shape));

  PADDLE_ENFORCE_EQ(
      y_shape.size(),
      2,
      phi::errors::InvalidArgument(
          "The dimension of y should be 2 but received y's shape: [%s]",
          y_shape));

  // Validate x's width equals y's height
  PADDLE_ENFORCE_EQ(x_shape[1],
                    y_shape[0],
                    phi::errors::InvalidArgument(
                        "The input Variable x's width must be equal with "
                        "Variable y's height. "
                        "But received x's shape = [%s], y's shape = [%s].",
                        x_shape,
                        y_shape));

  // Validate input shape broadcast compatibility
  if (input_shape.size() == 2) {
    ValidateBroadcastDim(input_shape[0],
                         x_shape[0],
                         "The dimension 0 of input must be equal to x's "
                         "dimension 0, or must be 1.");
    ValidateBroadcastDim(input_shape[1],
                         y_shape[1],
                         "The dimension 1 of input must be equal to y's "
                         "dimension 1, or must be 1.");
  } else if (input_shape.size() == 1) {
    ValidateBroadcastDim(input_shape[0],
                         y_shape[1],
                         "The dimension 0 of input must be equal to y's "
                         "dimension 1, or must be 1.");
  } else {
    PADDLE_THROW(
        phi::errors::InvalidArgument("The dimension of input should be 2 or 1 "
                                     "but received input's shape: [%ld].",
                                     input_shape.size()));
  }
}

// Addmm broadcast validation for static graph
void AddmmPreProcess(pir::Value* input, pir::Value* x, pir::Value* y) {
  auto input_shape = pir::GetShapeFromValue(*input);
  auto x_shape = pir::GetShapeFromValue(*x);
  auto y_shape = pir::GetShapeFromValue(*y);

  // Validate x and y are 2D
  PADDLE_ENFORCE_EQ(
      x_shape.size(),
      2,
      phi::errors::InvalidArgument(
          "The dimension of x should be 2 but received x's shape size: %d",
          x_shape.size()));

  PADDLE_ENFORCE_EQ(
      y_shape.size(),
      2,
      phi::errors::InvalidArgument(
          "The dimension of y should be 2 but received y's shape size: %d",
          y_shape.size()));

  // Validate x's width equals y's height
  PADDLE_ENFORCE_EQ(x_shape[1],
                    y_shape[0],
                    phi::errors::InvalidArgument(
                        "The input Variable x's width must be equal with "
                        "Variable y's height. "
                        "But received x's shape[1] = %d, y's shape[0] = %d.",
                        x_shape[1],
                        y_shape[0]));
  // Validate input shape broadcast compatibility
  if (input_shape.size() == 2) {
    ValidateBroadcastDim(input_shape[0],
                         x_shape[0],
                         "The dimension 0 of input must be equal to x's "
                         "dimension 0, or must be 1.");
    ValidateBroadcastDim(input_shape[1],
                         y_shape[1],
                         "The dimension 1 of input must be equal to y's "
                         "dimension 1, or must be 1.");
  } else if (input_shape.size() == 1) {
    ValidateBroadcastDim(input_shape[0],
                         y_shape[1],
                         "The dimension 0 of input must be equal to y's "
                         "dimension 1, or must be 1.");
  } else {
    PADDLE_THROW(
        phi::errors::InvalidArgument("The dimension of input should be 2 or 1 "
                                     "but received input's dimension: %ld.",
                                     input_shape.size()));
  }
}

// Baddbmm broadcast validation for dygraph
void BaddbmmPreProcess(Tensor* input, Tensor* x, Tensor* y) {
  auto input_shape = input->dims();
  auto x_shape = x->dims();
  auto y_shape = y->dims();

  // Validate x and y are 3D
  PADDLE_ENFORCE_EQ(
      x_shape.size(),
      3,
      phi::errors::InvalidArgument(
          "The dimension of x should be 3 but received x's shape size: %d.",
          x_shape.size()));

  PADDLE_ENFORCE_EQ(
      y_shape.size(),
      3,
      phi::errors::InvalidArgument(
          "The dimension of y should be 3 but received y's shape size: %d.",
          y_shape.size()));

  // Validate x's width equals y's height
  PADDLE_ENFORCE_EQ(x_shape[2],
                    y_shape[1],
                    phi::errors::InvalidArgument(
                        "The input Variable x's width must be equal with "
                        "Variable y's height. "
                        "But received x's shape[2] = %d, y's shape[1] = %d.",
                        x_shape[2],
                        y_shape[1]));

  // Validate input shape broadcast compatibility
  if (input_shape.size() == 3) {
    ValidateBroadcastDim(input_shape[0],
                         x_shape[0],
                         "The dimension 0 of input must be equal to x's "
                         "dimension 0, or must be 1.");
    ValidateBroadcastDim(input_shape[1],
                         x_shape[1],
                         "The dimension 1 of input must be equal to x's "
                         "dimension 1, or must be 1.");
    ValidateBroadcastDim(input_shape[2],
                         y_shape[2],
                         "The dimension 2 of input must be equal to y's "
                         "dimension 2, or must be 1.");
  } else if (input_shape.size() == 2) {
    ValidateBroadcastDim(input_shape[0],
                         x_shape[1],
                         "The dimension 0 of input must be equal to x's "
                         "dimension 1, or must be 1.");
    ValidateBroadcastDim(input_shape[1],
                         y_shape[2],
                         "The dimension 1 of input must be equal to y's "
                         "dimension 2, or must be 1.");
  } else {
    PADDLE_THROW(
        phi::errors::InvalidArgument("The dimension of input should be "
                                     "3 or 2 but received input's "
                                     "dimension: %ld.",
                                     input_shape.size()));
  }
}

// Baddbmm broadcast validation for static graph
void BaddbmmPreProcess(pir::Value* input, pir::Value* x, pir::Value* y) {
  auto input_shape = pir::GetShapeFromValue(*input);
  auto x_shape = pir::GetShapeFromValue(*x);
  auto y_shape = pir::GetShapeFromValue(*y);

  // Validate x and y are 3D
  PADDLE_ENFORCE_EQ(
      x_shape.size(),
      3,
      phi::errors::InvalidArgument(
          "The dimension of x should be 3 but received x's shape size: %d",
          x_shape.size()));

  PADDLE_ENFORCE_EQ(
      y_shape.size(),
      3,
      phi::errors::InvalidArgument(
          "The dimension of y should be 3 but received y's shape size: %d",
          y_shape.size()));

  // Validate x's width equals y's height
  PADDLE_ENFORCE_EQ(x_shape[2],
                    y_shape[1],
                    phi::errors::InvalidArgument(
                        "The input Variable x's width must be equal with "
                        "Variable y's height. "
                        "But received x's shape[2] = %d, y's shape[1] = %d.",
                        x_shape[2],
                        y_shape[1]));

  // Validate input shape broadcast compatibility
  if (input_shape.size() == 3) {
    ValidateBroadcastDim(input_shape[0],
                         x_shape[0],
                         "The dimension 0 of input must be equal to x's "
                         "dimension 0, or must be 1.");
    ValidateBroadcastDim(input_shape[1],
                         x_shape[1],
                         "The dimension 1 of input must be equal to x's "
                         "dimension 1, or must be 1.");
    ValidateBroadcastDim(input_shape[2],
                         y_shape[2],
                         "The dimension 2 of input must be equal to y's "
                         "dimension 2, or must be 1.");
  } else if (input_shape.size() == 2) {
    ValidateBroadcastDim(input_shape[0],
                         x_shape[1],
                         "The dimension 0 of input must be equal to x's "
                         "dimension 1, or must be 1.");
    ValidateBroadcastDim(input_shape[1],
                         y_shape[2],
                         "The dimension 1 of input must be equal to y's "
                         "dimension 2, or must be 1.");
  } else {
    PADDLE_THROW(
        phi::errors::InvalidArgument("The dimension of input should be "
                                     "3 or 2 but received input's "
                                     "dimension: %ld.",
                                     input_shape.size()));
  }
}

// Inplace API broadcast validation for dygraph
void InplaceShapePreProcess(Tensor* x, Tensor* y) {
  auto x_shape = x->dims();
  auto y_shape = y->dims();

  auto out_shape = phi::funcs::BroadcastTwoDims(x_shape, y_shape);

  PADDLE_ENFORCE_EQ(
      out_shape,
      x_shape,
      phi::errors::InvalidArgument(
          "The shape of broadcast output %s is different from that of inplace "
          "tensor %s in the Inplace operation.",
          out_shape,
          x_shape));
}

// Inplace API broadcast validation for static graph
void InplaceShapePreProcess(pir::Value* x, pir::Value* y) {
  auto x_shape = pir::GetShapeFromValue(*x);
  auto y_shape = pir::GetShapeFromValue(*y);

  auto out_shape = phi::funcs::BroadcastTwoDims(common::make_ddim(x_shape),
                                                common::make_ddim(y_shape));

  PADDLE_ENFORCE_EQ(
      out_shape,
      common::make_ddim(x_shape),
      phi::errors::InvalidArgument(
          "The shape of broadcast output %s is different from that of inplace "
          "tensor %s in the Inplace operation.",
          out_shape,
          common::make_ddim(x_shape)));
}

}  // namespace pybind

}  // namespace paddle
