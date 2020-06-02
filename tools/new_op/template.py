# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
__all__ = [
    'Template', 'op_class_template', 'copyright_template', 'header_template'
]


class Template(object):
    def __init__(self, template):
        self.template = template

    def format(self, **kwargs):
        for k in kwargs:
            key = '@' + k
            value = kwargs[k]
            rgx = re.compile(key)
            self.template = rgx.sub(value, self.template)
        return self.template


copyright_template = """
/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
"""

header_template = """
#pragma once

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/math_function.h"

"""

op_class_template = """
class @op_class_name : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    // step 1: check inputs and outputs 
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "Mul");
    OP_INOUT_CHECK(ctx->HasInput("Y"), "Input", "Y", "Mul");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "Mul");


    // step 2: do some check on input shape
    auto x_dims = ctx->GetInputDim("X");
    auto y_dims = ctx->GetInputDim("Y");

    // do some shape check here 
    // ref: https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/addon_development/new_op/new_op.html#infershape-compile-time-run-time

    // step 3: calculate and set output shape
    std::vector<int64_t> output_dims;
    // calculate output dims by input dims
    ctx->SetOutputDim("Out", framework::make_ddim(output_dims));
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const {
    framework::LibraryType library = framework::LibraryType::kPlain;
    framework::DataLayout layout = framework::DataLayout::kAnyLayout;
    int customized_type_value =
        framework::OpKernelType::kDefaultCustomizedTypeValue;
    auto input_data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
#ifdef PADDLE_WITH_MKLDNN
    if (library == framework::LibraryType::kPlain &&
        platform::CanMKLDNNBeUsed(ctx)) {
      library = framework::LibraryType::kMKLDNN;
      layout = framework::DataLayout::kMKLDNN;

      if (input_data_type == framework::DataTypeTrait<int8_t>::DataType() ||
          input_data_type == framework::DataTypeTrait<uint8_t>::DataType()) {
        customized_type_value = kMULMKLDNNINT8;
      }
    }
#endif

    return framework::OpKernelType(input_data_type, ctx.GetPlace(), layout,
                                   library, customized_type_value);
  }
};
"""

op_maker_template = """
class MulOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("@", "(Tensor), The first input tensor of mul op.");
    AddInput("Y", "(Tensor), The second input tensor of mul op.");
    AddOutput("Out", "(Tensor), The output tensor of mul op.");
    AddAttr<bool>("use_mkldnn",
                  "(bool, default false) Only used in mkldnn kernel")
        .SetDefault(false);
    AddAttr<int>(
        "x_num_col_dims",
        R"DOC((int, default 1), The mul_op can take tensors with more than two
              dimensions as its inputs. If the input $X$ is a tensor with more
              than two dimensions, $X$ will be flattened into a two-dimensional
              matrix first. The flattening rule is: the first `num_col_dims`
              will be flattened to form the first dimension of the final matrix
              (the height of the matrix), and the rest `rank(X) - num_col_dims`
              dimensions are flattened to form the second dimension of the final
              matrix (the width of the matrix). As a result, height of the
              flattened matrix is equal to the product of $X$'s first
              `x_num_col_dims` dimensions' sizes, and width of the flattened
              matrix is equal to the product of $X$'s last `rank(x) - num_col_dims`
              dimensions' size. For example, suppose $X$ is a 6-dimensional
              tensor with the shape [2, 3, 4, 5, 6], and `x_num_col_dims` = 3.
              Thus, the flattened matrix will have a shape [2 x 3 x 4, 5 x 6] =
              [24, 30].
        )DOC")
        .SetDefault(1)
        .EqualGreaterThan(1);
    AddAttr<int>(
        "y_num_col_dims",
        R"DOC((int, default 1), The mul_op can take tensors with more than two,
              dimensions as its inputs. If the input $Y$ is a tensor with more
              than two dimensions, $Y$ will be flattened into a two-dimensional
              matrix first. The attribute `y_num_col_dims` determines how $Y$ is
              flattened. See comments of `x_num_col_dims` for more details.
        )DOC")
        .SetDefault(1)
        .EqualGreaterThan(1);
    AddAttr<float>(
        "scale_x",
        "scale_x to be used for int8 mul input data x. scale_x has the"
        "same purpose as scale_in in OPs that support quantization."
        "Only to be used with MKL-DNN INT8")
        .SetDefault(1.0f);
    AddAttr<std::vector<float>>(
        "scale_y",
        "scale_y to be used for int8 mul input data y. scale_y has the"
        "same purpose as scale_weights in OPs that support quantization."
        "Only to be used with MKL-DNN INT8")
        .SetDefault({1.0f});
    AddAttr<float>("scale_out",
                   "scale_out to be used for int8 output data."
                   "Only used with MKL-DNN INT8")
        .SetDefault(1.0f);
    AddAttr<bool>(
        "force_fp32_output",
        "(bool, default false) Force quantize kernel output FP32, only "
        "used in quantized MKL-DNN.")
        .SetDefault(false);
    AddComment(R"DOC(
        // please add commments of the operator here
)DOC");
  }
};
"""

add_input_template = """
    AddInput("@", "(Tensor), The @ input tensor of mul op.");
"""

add_attr_template = """
"""

grad_op_class_template = """
class @grad_op_class_name : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "mul");
    OP_INOUT_CHECK(ctx->HasInput("Y"), "Input", "Y", "mul");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   "Out@GRAD", "mul");
    auto x_dims = ctx->GetInputDim("X");
    auto y_dims = ctx->GetInputDim("Y");

    auto x_grad_name = framework::GradVarName("X");
    auto y_grad_name = framework::GradVarName("Y");

    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, x_dims);
    }
    if (ctx->HasOutput(y_grad_name)) {
      ctx->SetOutputDim(y_grad_name, y_dims);
    }
  }
};
"""

if __name__ == "__main__":
    # if len(sys.argv) > 1 and sys.argv[1].isdigit():
    #     check_approval(int(sys.argv[1]), sys.argv[2:])
    # else:
    #     print(
    #         "Usage: python check_pr_approval.py [count] [required reviewer id] ..."
    #     )
    t = Template(class_template)
    print(t.format(op_name='MulOp'))
