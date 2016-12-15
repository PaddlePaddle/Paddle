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

namespace paddle {

// NCHW
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
void CrossMapNormalGrad<DEVICE_TYPE_CPU>::operator()(CpuMatrix& inputsGrad,
                                                     CpuMatrix& inputsValue,
                                                     CpuMatrix& outputsGrad,
                                                     CpuMatrix& outputsValue,
                                                     CpuMatrix& denoms,
                                                     size_t channels,
                                                     size_t imgSizeH,
                                                     size_t imgSizeW,
                                                     size_t sizeX,
                                                     real scale,
                                                     real pow) {
  CHECK(inputsGrad.isContiguous());
  CHECK(outputsGrad.isContiguous());
  CHECK(denoms.isContiguous());
  CHECK(inputsValue.isContiguous());
  CHECK(outputsValue.isContiguous());
  CHECK_EQ(inputsGrad.getHeight(), outputsGrad.getHeight());
  CHECK_EQ(inputsGrad.getWidth(), outputsGrad.getWidth());
  CHECK_EQ(inputsGrad.getHeight(), denoms.getHeight());
  CHECK_EQ(inputsGrad.getWidth(), denoms.getWidth());
  CHECK_EQ(inputsGrad.getHeight(), inputsValue.getHeight());
  CHECK_EQ(inputsGrad.getWidth(), inputsValue.getWidth());
  CHECK_EQ(inputsGrad.getHeight(), outputsValue.getHeight());
  CHECK_EQ(inputsGrad.getWidth(), outputsValue.getWidth());

  size_t numSample = inputsGrad.getHeight();
  size_t numCols = inputsGrad.getWidth();
  size_t imageSize = imgSizeH * imgSizeW;
  CHECK(imageSize * channels == numCols);

  std::function<CpuVector(real*, size_t)> oneImage = [=](real* data,
                                                         size_t offset) {
    return CpuVector(imageSize, data + offset);
  };

  const int start = -((int)sizeX) / 2;
  const int end = (int)sizeX + start;
  const real ratio = -(real)2 * scale * pow;
  for (size_t i = 0; i < numSample; i++) {
    size_t sOffset = i * numCols;
    real* inputGradData = inputsGrad.getData() + sOffset;
    real* inputData = inputsValue.getData() + sOffset;
    real* denomData = denoms.getData() + sOffset;
    real* outputGradData = outputsGrad.getData() + sOffset;
    real* outputData = outputsValue.getData() + sOffset;

    for (int c = 0; c < (int)channels; c++) {
      size_t cOffset = c * imageSize;
      CpuVector inputGrad = oneImage(inputGradData, cOffset);
      CpuVector inputValue = oneImage(inputData, cOffset);
      CpuVector denom = oneImage(denomData, cOffset);
      CpuVector outputGrad = oneImage(outputGradData, cOffset);

      inputGrad = inputGrad + denom.pow(-pow) * outputGrad;
      for (int s = start; s < end; s++) {
        if (c + s >= 0 && c + s < (int)channels) {
          size_t offset = (c + s) * imageSize;
          CpuVector output = oneImage(outputData, offset);
          CpuVector outputGrad = oneImage(outputGradData, offset);
          CpuVector denom = oneImage(denomData, offset);

          inputGrad += ((outputGrad * output * ratio) / denom) * inputValue;
        }
      }
    }
  }
}

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

REGISTER_TYPED_FUNC(CrossMapNormal, CPU, CrossMapNormalFunc);
REGISTER_TYPED_FUNC(CrossMapNormal, GPU, CrossMapNormalFunc);

}  // namespace paddle
