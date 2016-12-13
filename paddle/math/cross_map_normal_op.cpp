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
void CrossMapNormal::operator()(CpuMatrix& outputs,
                                CpuMatrix& denoms,
                                CpuMatrix& inputs,
                                size_t channels,
                                size_t imgSizeH,
                                size_t imgSizeW,
                                size_t sizeX,
                                real scale,
                                real pow) {
  CHECK(outputs.isContiguous());
  CHECK(inputs.isContiguous());
  CHECK(denoms.isContiguous());
  CHECK_EQ(outputs.getHeight(), inputs.getHeight());
  CHECK_EQ(outputs.getWidth(), inputs.getWidth());
  CHECK_EQ(outputs.getHeight(), denoms.getHeight());
  CHECK_EQ(outputs.getWidth(), denoms.getWidth());

  size_t numSample = inputs.getHeight();
  size_t numCols = inputs.getWidth();
  size_t imageSize = imgSizeH * imgSizeW;
  CHECK(imageSize * channels == numCols);

  denoms = denoms.constant(1.0);
  const int start = -((int)sizeX - 1) / 2;
  const int end = (int)sizeX + start;
  for (size_t i = 0; i < numSample; i++) {
    real* denomsData = denoms.getData() + i * numCols;
    real* inputData = inputs.getData() + i * numCols;
    for (int c = 0; c < (int)channels; c++) {
      CpuVector denom(imageSize, denomsData + c * imageSize);
      for (int s = start; s < end; s++) {
        if (c + s >= 0 && c + s < (int)channels) {
          CpuVector input(imageSize, inputData + (c + s) * imageSize);
          denom += input.square() * scale;
        }
      }
    }
  }
  outputs = inputs * denoms.pow(-pow);
}

void CrossMapNormalGrad::operator()(CpuMatrix& inputsGrad,
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

}  // namespace paddle
