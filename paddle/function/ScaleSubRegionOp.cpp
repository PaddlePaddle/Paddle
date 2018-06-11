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

#include "ScaleSubRegionOp.h"
#include "paddle/function/TensorShape.h"

namespace paddle {

template <>
void ScaleSubRegion<DEVICE_TYPE_CPU>(real* outputs,
                                     const real* inputs,
                                     const real* indices,
                                     const TensorShape shape,
                                     const FuncConfig& conf) {
  real value = conf.get<real>("value");

  int number = shape[0];
  int channel = shape[1];
  int height = shape[2];
  int width = shape[3];

  memcpy(outputs, inputs, number * channel * height * width * sizeof(real));

  for (int n = 0; n < number; ++n) {
    // indices start from 1
    int offset = n * 6;
    for (int c = indices[offset] - 1; c < indices[offset + 1]; ++c) {
      for (int h = indices[offset + 2] - 1; h < indices[offset + 3]; ++h) {
        for (int w = indices[offset + 4] - 1; w < indices[offset + 5]; ++w) {
          int idx = ((n * channel + c) * height + h) * width + w;
          outputs[idx] *= value;
        }
      }
    }
  }
}

template <>
void ScaleSubRegionGrad<DEVICE_TYPE_CPU>(const real* inGrad,
                                         real* outGrad,
                                         const real* indices,
                                         const TensorShape shape,
                                         const FuncConfig& conf) {
  real value = conf.get<real>("value");

  int number = shape[0];
  int channel = shape[1];
  int height = shape[2];
  int width = shape[3];

  for (int n = 0; n < number; ++n) {
    for (int c = 0; c < channel; ++c) {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          int idx = ((n * channel + c) * height + h) * width + w;
          int offset = n * 6;
          if (c >= (indices[offset] - 1) && c <= (indices[offset + 1] - 1) &&
              h >= (indices[offset + 2] - 1) &&
              h <= (indices[offset + 3] - 1) &&
              w >= (indices[offset + 4] - 1) &&
              w <= (indices[offset + 5] - 1)) {
            outGrad[idx] += inGrad[idx] * value;
          } else {
            outGrad[idx] += inGrad[idx];
          }
        }
      }
    }
  }
}

/**
 * \brief For each instance, ScaleSubRegion can be used to multiply a value to
 *        a specified sub continuous region. By providing start index and end
 *        index for C/H/W, you can specify the location and shape of the region.
 *
 * Argument in this Function:
 * \param inputs    A 4-D tensor with shape [N, C, H, W], only one input.
 * \param indices   A 2-D tensor with shape [N, 6], indicates the sub region.
 * \param outputs   A 4-D tensor with same shape as inputs, output value.
 */
template <DeviceType Device>
class ScaleSubRegionFunc : public FunctionBase {
 public:
  void init(const FuncConfig& config) override { conf_ = config; }

  void calc(const BufferArgs& inputs, const BufferArgs& outputs) override {
    CHECK_EQ(2UL, inputs.size());
    CHECK_EQ(1UL, outputs.size());
    CHECK_EQ(outputs[0].getArgType(), ASSIGN_TO);

    TensorShape shape = inputs[0].shape();

    ScaleSubRegion<Device>(outputs[0].data<real>(),
                           inputs[0].data<real>(),
                           inputs[1].data<real>(),
                           shape,
                           conf_);
  }

 private:
  FuncConfig conf_;
};

/**
 * \brief The backward propagation of ScaleSubRegion Function.
 *
 * Argument in this Function:
 * \param inputs  A 4-D tensor with shape [N, C, H, W], output gradient.
 * \param indices A 2-D tensor with shape [N, 6], indicates the sub region.
 * \param outputs A 4-D tensor with shape [N, C, H, W], gradient of input value.
 */

template <DeviceType Device>
class ScaleSubRegionGradFunc : public FunctionBase {
 public:
  void init(const FuncConfig& config) override { conf_ = config; }

  void calc(const BufferArgs& inputs, const BufferArgs& outputs) override {
    CHECK_EQ(2UL, inputs.size());
    CHECK_EQ(1UL, outputs.size());
    CHECK_EQ(outputs[0].getArgType(), ADD_TO);

    TensorShape shape = inputs[0].shape();

    ScaleSubRegionGrad<Device>(inputs[0].data<real>(),
                               outputs[0].data<real>(),
                               inputs[1].data<real>(),
                               shape,
                               conf_);
  }

 private:
  FuncConfig conf_;
};

REGISTER_TYPED_FUNC(ScaleSubRegion, CPU, ScaleSubRegionFunc);
REGISTER_TYPED_FUNC(ScaleSubRegionGrad, CPU, ScaleSubRegionGradFunc);
#ifdef PADDLE_WITH_CUDA
REGISTER_TYPED_FUNC(ScaleSubRegion, GPU, ScaleSubRegionFunc);
REGISTER_TYPED_FUNC(ScaleSubRegionGrad, GPU, ScaleSubRegionGradFunc);
#endif

}  // namespace paddle
