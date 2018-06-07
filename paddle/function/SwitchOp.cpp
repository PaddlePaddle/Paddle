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

#include "SwitchOp.h"
#include "paddle/math/Vector.h"

namespace paddle {

template <>
void NCHW2NHWC<DEVICE_TYPE_CPU>(real* outputs,
                                const real* inputs,
                                const int num,
                                const int inC,
                                const int inH,
                                const int inW,
                                const int argType) {
  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < inC; ++c) {
      for (int h = 0; h < inH; ++h) {
        for (int w = 0; w < inW; ++w) {
          if (argType == ADD_TO) {
            outputs[((n * inH + h) * inW + w) * inC + c] += *(inputs++);
          } else {
            outputs[((n * inH + h) * inW + w) * inC + c] = *(inputs++);
          }
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
                                const int inC,
                                const int argType) {
  for (int n = 0; n < num; ++n) {
    for (int h = 0; h < inH; ++h) {
      for (int w = 0; w < inW; ++w) {
        for (int c = 0; c < inC; ++c) {
          if (argType == ADD_TO) {
            outputs[((n * inC + c) * inH + h) * inW + w] += *(inputs++);
          } else {
            outputs[((n * inC + c) * inH + h) * inW + w] = *(inputs++);
          }
        }
      }
    }
  }
}

/**
 * \brief  Switch dimension order of image input.
 *         The input and output is a 4D tensor. Switch order
 *         'batch_size,channels, height, width' to
 *         order 'batch_size, height, width, channels'.
 *
 * Argument in this Function:
 * \param inputs  input data with order 'batch_size,channels, height, width'.
 * \param outputs output data with order 'batch_size, height, width, channels'.
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
    NCHW2NHWC<Device>(outputs[0].data<real>(),
                      inputs[0].data<real>(),
                      num,
                      inC,
                      inH,
                      inW,
                      outputs[0].getArgType());
  }
};

/**
 * \brief  Switch dimension order of image input.
 *         The input and output is a 4D tensor. Switch order
 *         'batch_size, height, width, channels' to
 *         order 'batch_size, channels, height, width'.
 *
 * Argument in this Function:
 * \param inputs  input data with order 'batch_size, height, width, channels'.
 * \param outputs output data with order 'batch_size, channels, height, width'.
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

    NHWC2NCHW<Device>(outputs[0].data<real>(),
                      inputs[0].data<real>(),
                      num,
                      inH,
                      inW,
                      inC,
                      outputs[0].getArgType());
  }
};

REGISTER_TYPED_FUNC(NCHW2NHWC, CPU, NCHW2NHWCFunc);
REGISTER_TYPED_FUNC(NHWC2NCHW, CPU, NHWC2NCHWFunc);
#ifdef PADDLE_WITH_CUDA
REGISTER_TYPED_FUNC(NCHW2NHWC, GPU, NCHW2NHWCFunc);
REGISTER_TYPED_FUNC(NHWC2NCHW, GPU, NHWC2NCHWFunc);
#endif

}  // namespace paddle
