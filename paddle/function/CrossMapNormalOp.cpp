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

#include "CrossMapNormalOp.h"
#include "paddle/math/Vector.h"

namespace paddle {

template <>
void CrossMapNormal<DEVICE_TYPE_CPU>(real* outputs,
                                     real* denoms,
                                     const real* inputs,
                                     size_t numSamples,
                                     size_t channels,
                                     size_t height,
                                     size_t width,
                                     size_t size,
                                     real scale,
                                     real pow) {
  size_t oneImage = height * width;
  size_t oneSample = channels * oneImage;

  CpuVector outputsV(numSamples * oneSample, outputs);
  CpuVector inputsV(numSamples * oneSample, const_cast<real*>(inputs));
  CpuVector denomsV(numSamples * oneSample, denoms);

  // f(x) = x * ( 1 + scale * SUM((x)^2) )^(-pow)
  // x represents inputs
  // f(x) represents outputs
  // denoms save the intermediate result for backward
  denomsV = denomsV.constant(1.0);
  const int start = -((int)size - 1) / 2;
  const int end = (int)size + start;
  for (size_t i = 0; i < numSamples; i++) {
    real* oneDenom = denoms + i * oneSample;
    real* oneInput = const_cast<real*>(inputs) + i * oneSample;
    for (int c = 0; c < (int)channels; c++) {
      CpuVector denom(oneImage, oneDenom + c * oneImage);
      for (int s = start; s < end; s++) {
        if (c + s >= 0 && c + s < (int)channels) {
          CpuVector input(oneImage, oneInput + (c + s) * oneImage);
          denom += input.square() * scale;
        }
      }
    }
  }

  outputsV = inputsV * denomsV.pow(-pow);
}

template <>
void CrossMapNormalGrad<DEVICE_TYPE_CPU>(real* inputsGrad,
                                         const real* inputsValue,
                                         const real* outputsValue,
                                         const real* outputsGrad,
                                         const real* denoms,
                                         size_t numSamples,
                                         size_t channels,
                                         size_t height,
                                         size_t width,
                                         size_t size,
                                         real scale,
                                         real pow) {
  size_t oneSample = channels * height * width;
  std::function<CpuVector(real*, size_t)> oneImage = [=](real* data,
                                                         size_t offset) {
    return CpuVector(height * width, data + offset);
  };

  const int start = -((int)size) / 2;
  const int end = (int)size + start;
  const real ratio = -(real)2 * scale * pow;
  for (size_t i = 0; i < numSamples; i++) {
    size_t sOffset = i * oneSample;
    real* oneInputGrad = inputsGrad + sOffset;
    real* oneInputValue = const_cast<real*>(inputsValue) + sOffset;
    real* oneDenom = const_cast<real*>(denoms) + sOffset;
    real* oneOutputGrad = const_cast<real*>(outputsGrad) + sOffset;
    real* oneOutputValue = const_cast<real*>(outputsValue) + sOffset;

    for (int c = 0; c < (int)channels; c++) {
      size_t cOffset = c * height * width;
      CpuVector inputGrad = oneImage(oneInputGrad, cOffset);
      CpuVector inputValue = oneImage(oneInputValue, cOffset);
      CpuVector denom = oneImage(oneDenom, cOffset);
      CpuVector outputGrad = oneImage(oneOutputGrad, cOffset);

      inputGrad = inputGrad + denom.pow(-pow) * outputGrad;
      for (int s = start; s < end; s++) {
        if (c + s >= 0 && c + s < (int)channels) {
          size_t offset = (c + s) * height * width;
          CpuVector output = oneImage(oneOutputValue, offset);
          CpuVector outputGrad = oneImage(oneOutputGrad, offset);
          CpuVector denom = oneImage(oneDenom, offset);

          inputGrad += ((outputGrad * output * ratio) / denom) * inputValue;
        }
      }
    }
  }
}

/**
 * \brief {o_0, o_1} = calc(i_0)
 *
 * \param inputs[0] input value.
 * \param outputs[0] output value.
 * \param outputs[1] denoms.
 */
template <DeviceType Device>
class CrossMapNormalFunc : public FunctionBase {
public:
  void init(const FuncConfig& config) override {
    size_ = config.get<size_t>("size");
    scale_ = config.get<real>("scale");
    pow_ = config.get<real>("pow");
  }

  void calc(const BufferArgs& inputs, const BufferArgs& outputs) override {
    CHECK_EQ((size_t)1, inputs.size());
    CHECK_EQ((size_t)2, outputs.size());

    CHECK_EQ(inputs[0].shape().ndims(), (size_t)4);
    CHECK(inputs[0].shape() == outputs[0].shape());
    CHECK(inputs[0].shape() == outputs[1].shape());

    CHECK_EQ(outputs[0].getArgType(), ASSIGN_TO);
    CHECK_EQ(outputs[1].getArgType(), ASSIGN_TO);
    size_t samples = inputs[0].shape()[0];
    size_t channels = inputs[0].shape()[1];
    size_t height = inputs[0].shape()[2];
    size_t width = inputs[0].shape()[3];

    CrossMapNormal<Device>(outputs[0].data<real>(),
                           outputs[1].data<real>(),
                           inputs[0].data<real>(),
                           samples,
                           channels,
                           height,
                           width,
                           size_,
                           scale_,
                           pow_);
  }

private:
  size_t size_;
  real scale_;
  real pow_;
};

/**
 * \brief {o_0} = calc(i_0, i_1, i_2, i_3)
 *
 * \param inputs[0] input value.
 * \param inputs[1] output value.
 * \param inputs[2] output grad.
 * \param inputs[3] denoms.
 * \param outputs[0] input grad.
 */
template <DeviceType Device>
class CrossMapNormalGradFunc : public FunctionBase {
public:
  void init(const FuncConfig& config) override {
    size_ = config.get<size_t>("size");
    scale_ = config.get<real>("scale");
    pow_ = config.get<real>("pow");
  }

  void calc(const BufferArgs& inputs, const BufferArgs& outputs) override {
    CHECK_EQ((size_t)4, inputs.size());
    CHECK_EQ((size_t)1, outputs.size());

    CHECK_EQ(inputs[0].shape().ndims(), (size_t)4);
    CHECK(inputs[0].shape() == inputs[1].shape());
    CHECK(inputs[0].shape() == inputs[2].shape());
    CHECK(inputs[0].shape() == inputs[3].shape());
    CHECK(inputs[0].shape() == outputs[0].shape());

    // TODO(hedaoyuan): need support ASSIGN_TO mode.
    CHECK_EQ(outputs[0].getArgType(), ADD_TO);

    size_t samples = inputs[0].shape()[0];
    size_t channels = inputs[0].shape()[1];
    size_t height = inputs[0].shape()[2];
    size_t width = inputs[0].shape()[3];

    CrossMapNormalGrad<Device>(outputs[0].data<real>(),
                               inputs[0].data<real>(),
                               inputs[1].data<real>(),
                               inputs[2].data<real>(),
                               inputs[3].data<real>(),
                               samples,
                               channels,
                               height,
                               width,
                               size_,
                               scale_,
                               pow_);
  }

private:
  size_t size_;
  real scale_;
  real pow_;
};

REGISTER_TYPED_FUNC(CrossMapNormal, CPU, CrossMapNormalFunc);
REGISTER_TYPED_FUNC(CrossMapNormalGrad, CPU, CrossMapNormalGradFunc);
#ifndef PADDLE_ONLY_CPU
REGISTER_TYPED_FUNC(CrossMapNormal, GPU, CrossMapNormalFunc);
REGISTER_TYPED_FUNC(CrossMapNormalGrad, GPU, CrossMapNormalGradFunc);
#endif

}  // namespace paddle
