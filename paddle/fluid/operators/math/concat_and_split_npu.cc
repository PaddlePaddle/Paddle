/* Copyright (c) 2021 paddlepaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef PADDLE_WITH_ASCEND_CL
#include "paddle/fluid/operators/math/concat_and_split.h"
#include "paddle/fluid/operators/concat_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace framework {
class Tensor;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace operators {
namespace math {

/*
 * All tensors' dimension should be the same and the values of
 * each dimension must be the same, except the axis dimension.
 */
template <typename T>
class ConcatFunctor<platform::NPUDeviceContext, T> {
 public:
  void operator()(const platform::NPUDeviceContext& context,
                  const std::vector<framework::Tensor>& input, int axis,
                  framework::Tensor* output) {
    PADDLE_ENFORCE_GT(input.size(), 0,
                      platform::errors::InvalidArgument(
                          "The input tensor size shoule be greater than 0."));
    axis = ComputeAxis(static_cast<int64_t>(axis),
                       static_cast<int64_t>(input[0].dims().size()));
    auto place = context.GetPlace();
    output->mutable_data<T>(place);

    std::vector<framework::Tensor> inputs;
    std::vector<std::string> names;
    for (size_t i = 0; i < input.size(); i++) {
      if (input[i].numel() > 0) {
        inputs.push_back(input[i]);
        names.push_back("x" + std::to_string(i));
      } else {
        continue;
      }
    }

    auto stream = context.stream();
    NpuOpRunner runner{
        "ConcatD",
        {inputs},
        {*output},
        {{"concat_dim", axis}, {"N", static_cast<int>(inputs.size())}}};
    runner.AddInputNames(names);
    runner.Run(stream);
  }
};

template <typename T>
class SplitFunctor<platform::NPUDeviceContext, T> {
 public:
  void operator()(const platform::NPUDeviceContext& context,
                  const framework::Tensor& input,
                  const std::vector<const framework::Tensor*>& ref_inputs,
                  const int axis, std::vector<framework::Tensor*>* outputs) {
    PADDLE_ENFORCE_NE(
        input.numel(), 0,
        platform::errors::InvalidArgument(
            "The input tensor has zero element which is invalid."));
    PADDLE_ENFORCE_EQ(
        ref_inputs.size(), outputs->size(),
        platform::errors::InvalidArgument(
            "The size of ref_inputs should be equal to the size of outputs"));
    PADDLE_ENFORCE_GT(
        outputs->size(), 0,
        platform::errors::InvalidArgument(
            "The size of output tensor list shoule be greater than 0."));
    PADDLE_ENFORCE_LT(
        axis, input.dims().size(),
        platform::errors::InvalidArgument(
            "The axis should not exceed the rank of input tensor."));

    std::vector<int> sections;
    int num = outputs->size();
    for (int i = 0; i < num; i++) {
      auto dim = ref_inputs[i]->dims();
      sections.push_back(dim[axis]);
    }

    std::vector<framework::Tensor> outputs_vec;
    auto place = context.GetPlace();
    for (int i = 0; i < num; i++) {
      outputs->at(i)->mutable_data<T>(place);
      outputs_vec.push_back(*(outputs->at(i)));
    }

    auto stream = context.stream();
    NpuOpRunner runner{"SplitVD",
                       {input},
                       {outputs_vec},
                       {{"size_splits", sections},
                        {"split_dim", axis},
                        {"num_split", static_cast<int32_t>(sections.size())}}};
    runner.Run(stream);
  }
};

#define DEFINE_NPU_FUNCTOR(type)                                  \
  template class ConcatFunctor<platform::NPUDeviceContext, type>; \
  template class SplitFunctor<platform::NPUDeviceContext, type>;

// NOTE(liubo48): test type float first, then add others.
DEFINE_NPU_FUNCTOR(float)

}  // namespace math
}  // namespace operators
}  // namespace paddle

#endif
