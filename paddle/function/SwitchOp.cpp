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

#include "SwitchOp.h"
#include "paddle/math/Vector.h"

namespace paddle {

template <>
void NCHW2NHWC<DEVICE_TYPE_CPU>(real* outputs,
                                const real* inputs,
                                const int num,
                                const int inC,
                                const int inH,
                                const int inW) {
  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < inC; ++c) {
      for (int h = 0; h < inH; ++h) {
        for (int w = 0; w < inW; ++w) {
          outputs[((n * inH + h) * inW + w) * inC + c] = *(inputs++);
        }
      }
    }
  }
}

template <>
void NHWC2NCHW<DEVICE_TYPE_CPU>(real* outputs,
                                const real* inputs,
                                const int num,
                                const int inH,
                                const int inW,
                                const int inC) {
  for (int n = 0; n < num; ++n) {
    for (int h = 0; h < inH; ++h) {
      for (int w = 0; w < inW; ++w) {
        for (int c = 0; c < inC; ++c) {
          outputs[((n * inC + c) * inH + h) * inW + w] = *(inputs++);
        }
      }
    }
  }
}

/**
 * \brief Padding zeros to input according to the specify dimension.
 *        The struct pad_ contains the padding size in each dimension.
 *        The input and output is a 4D tensor. In PadFunc, we only
 *        pad zeros to the 2nd to 4th dimension.
 *
 * Argument in this Function:
 * \param pad_    A struct object contains the padding size in each dimension.
 *                It has six integers. The channelStart and channelEnd indicate
 *                how many zeros to add before and after the input in channel
 *                dimension. And the heightStart and heightEnd indicate padding
 *                in height dimension. The widthStart and widthEnd indicate the
 *                padding in width dimension.
 * \param inputs  A 4D tensor, only one input.
 * \param outputs A 4D tensor, the output value after padding.
 *
 */

template <DeviceType Device>
class NCHW2NHWCFunc : public FunctionBase {
public:
  void init(const FuncConfig& config) override {}

  void calc(const BufferArgs& inputs, const BufferArgs& outputs) override {
    CHECK_EQ(1UL, inputs.size());
    CHECK_EQ(1UL, outputs.size());

    size_t num = inputs[0].shape()[0];
    size_t inC = inputs[0].shape()[1];
    size_t inH = inputs[0].shape()[2];
    size_t inW = inputs[0].shape()[3];
    typename Tensor<real, Device>::Vector vec(outputs[0].shape().getElements(),
                                              outputs[0].data<real>());
    vec.zero();

    NCHW2NHWC<Device>(
        outputs[0].data<real>(), inputs[0].data<real>(), num, inC, inH, inW);
  }
};

/**
 * \brief The backward propagation of padding Function. Remove the elements
 *        in the padding positions of forward.
 *
 * Argument in this Function:
 * \param pad_    The same meaning as it in PadFunc.
 * \param inputs  The gradient with respect to the output value of PadFunc.
 * \param outputs The gradient with respect to the input value of PadFunc.
 */

template <DeviceType Device>
class NHWC2NCHWFunc : public FunctionBase {
public:
  void init(const FuncConfig& config) override {}

  void calc(const BufferArgs& inputs, const BufferArgs& outputs) override {
    CHECK_EQ(1UL, inputs.size());
    CHECK_EQ(1UL, outputs.size());

    size_t num = inputs[0].shape()[0];
    size_t inH = inputs[0].shape()[1];
    size_t inW = inputs[0].shape()[2];
    size_t inC = inputs[0].shape()[3];

    NHWC2NCHW<Device>(
        outputs[0].data<real>(), inputs[0].data<real>(), num, inH, inW, inC);
  }
};

REGISTER_TYPED_FUNC(NCHW2NHWC, CPU, NCHW2NHWCFunc);
REGISTER_TYPED_FUNC(NHWC2NCHW, CPU, NHWC2NCHWFunc);
#ifndef PADDLE_ONLY_CPU
REGISTER_TYPED_FUNC(NCHW2NHWC, GPU, NCHW2NHWCFunc);
REGISTER_TYPED_FUNC(NHWC2NCHW, GPU, NHWC2NCHWFunc);
#endif

}  // namespace paddle
