/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "Function.h"

namespace paddle {

/*
 * Function Arguments:
 *
 * \param inputs[0]  Input image data, is NCHW format, where N is batch size,
 *                   C is the number of channels, H and W is the height and
 *                   width of input image.
 * \param inputs[1]  Filter data, is MCHW, where M is the number of output
 *                   channels, C is the number of input channels, H and W
 *                   is height and width of filter.
 * \param outputs[0] Output image data, is NCHW format, where N is batch size,
 *                   C is the number of channels, H and W is the height and
 *                   width of output image.
 *
 * \note Implemented based on the ConvFunctionBase class only supports
 *       input data in the NCHW format.
 */
class ConvFunctionBase : public FunctionBase {
public:
  void init(const FuncConfig& config) override {
    // function arguments
    stride_ = config.get<size_t>("stride");
    padding_ = config.get<size_t>("padding");

    // number of inputs and outputs
    numInputs_ = 2;
    numOutputs_ = 1;
  }

  virtual void calc(const BufferArgs& inputs, const BufferArgs& outputs) {}

  void check(const BufferArgs& inputs, const BufferArgs& outputs) override {
    CHECK_EQ(numInputs_, inputs.size());
    CHECK_EQ(numOutputs_, outputs.size());

    CHECK_EQ(inputs[0].shape().ndims(), (size_t)4);
    CHECK_EQ(inputs[1].shape().ndims(), (size_t)4);
    CHECK_EQ(outputs[0].shape().ndims(), (size_t)4);

    CHECK(inputs[0].shape()[0] == outputs[0].shape()[0]);
    CHECK(inputs[0].shape()[1] == inputs[1].shape()[1]);
    CHECK(outputs[0].shape()[1] == inputs[1].shape()[0]);
  }

protected:
  size_t padding_;
  size_t stride_;
};

}  // namespace paddle
