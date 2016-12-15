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

#include "cross_map_normal_op.h"
#include "paddle/math/Vector.h"

namespace paddle {

template <>
void CrossMapNormal<DEVICE_TYPE_CPU>(real* outputs,
                                     real* denoms,
                                     real* inputs,
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
  CpuVector inputsV(numSamples * oneSample, inputs);
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
    real* oneInput = inputs + i * oneSample;
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
                                         real* inputsValue,
                                         real* outputsValue,
                                         real* outputsGrad,
                                         real* denoms,
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
    real* oneInputValue = inputsValue + sOffset;
    real* oneDenom = denoms + sOffset;
    real* oneOutputGrad = outputsGrad + sOffset;
    real* oneOutputValue = outputsValue + sOffset;

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

  void calc(const Arguments& inputs,
            const Arguments& outputs,
            const Arguments& inouts) override {
    CHECK_EQ(1, inputs.size());
    CHECK_EQ(2, outputs.size());
    CHECK_EQ(0, inouts.size());

    CHECK_EQ(inputs[0].dims_.size(), 4);
    for (size_t i = 0; i < inputs[0].dims_.size(); i++) {
      CHECK_EQ(inputs[0].dims_[i], outputs[0].dims_[i]);
      CHECK_EQ(inputs[0].dims_[i], outputs[1].dims_[i]);
    }

    size_t samples = inputs[0].dims_[0];
    size_t channels = inputs[0].dims_[1];
    size_t height = inputs[0].dims_[2];
    size_t width = inputs[0].dims_[3];

    CrossMapNormal<Device>(outputs[0].getData(),
                           outputs[1].getData(),
                           inputs[0].getData(),
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

  void calc(const Arguments& inputs,
            const Arguments& outputs,
            const Arguments& inouts) override {
    CHECK_EQ(4, inputs.size());
    CHECK_EQ(1, outputs.size());
    CHECK_EQ(0, inouts.size());

    CHECK_EQ(inputs[0].dims_.size(), 4);
    for (size_t i = 0; i < inputs[0].dims_.size(); i++) {
      CHECK_EQ(inputs[0].dims_[i], inputs[1].dims_[i]);
      CHECK_EQ(inputs[0].dims_[i], inputs[2].dims_[i]);
      CHECK_EQ(inputs[0].dims_[i], inputs[3].dims_[i]);
      CHECK_EQ(inputs[0].dims_[i], outputs[0].dims_[i]);
    }

    size_t samples = inputs[0].dims_[0];
    size_t channels = inputs[0].dims_[1];
    size_t height = inputs[0].dims_[2];
    size_t width = inputs[0].dims_[3];

    CrossMapNormalGrad<Device>(outputs[0].getData(),
                               inputs[0].getData(),
                               inputs[1].getData(),
                               inputs[2].getData(),
                               inputs[3].getData(),
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
