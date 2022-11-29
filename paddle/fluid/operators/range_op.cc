/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/range_op.h"

#include <string>

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/ternary.h"

namespace paddle {
namespace operators {

class RangeOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetKernelTypeForVar(
      const std::string &var_name,
      const phi::DenseTensor &tensor,
      const framework::OpKernelType &expected_kernel_type) const override {
    if (platform::is_xpu_place(tensor.place())) {
      return framework::OpKernelType(
          expected_kernel_type.data_type_, tensor.place(), tensor.layout());
    }
    return expected_kernel_type;
  }
};

class RangeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Start",
             "Start of interval. The interval includes this value. It is a "
             "tensor with shape=[1].");
    AddInput("End",
             "End of interval. The interval does not include this value, "
             "except in some cases where step is not an integer and floating "
             "point round-off affects the length of out. It is a tensor with "
             "shape=[1].");
    AddInput("Step", "Spacing between values. It is a tensor with shape=[1].");
    AddOutput("Out", "A sequence of numbers.");
    AddComment(R"DOC(
    Return evenly spaced values within a given interval. Values are generated within the half-open interval [start, stop) (in other words, the interval including start but excluding stop). Like arange function of numpy.
)DOC");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(range,
                            RangeInferMetaFunctor,
                            PD_INFER_META(phi::ArangeInferMeta));
REGISTER_OP_WITHOUT_GRADIENT(range,
                             ops::RangeOp,
                             ops::RangeOpMaker,
                             RangeInferMetaFunctor);
