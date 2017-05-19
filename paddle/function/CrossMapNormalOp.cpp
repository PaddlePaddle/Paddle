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
#include "Register.h"
#include "paddle/math/Vector.h"
#include "paddle/topology/Attribute.h"

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

struct CrossMapNormalAttribute : public topology::Attribute {
  size_t size;
  double scale;
  double pow;

  REGISTER_FUNC_ATTRIBUTE() {
    regAttr(&CrossMapNormalAttribute::size, "size", "represent N in formula")
        .mustSet()
        .largerThan(0);

    regAttr(
        &CrossMapNormalAttribute::scale, "scale", "represent alpha in formula")
        .defaultValue(1.0)
        .largerThan(0.0);

    regAttr(&CrossMapNormalAttribute::pow, "pow", "represent beta in formula")
        .mustSet();
  }
};

template <DeviceType Device>
static Error forward(const BufferArgs& ins,
                     const BufferArgs& outs,
                     const CrossMapNormalAttribute& attrs) {
  size_t batchSize = ins[0].shape()[0];
  size_t maps = ins[0].shape()[1];
  size_t rows = ins[0].shape()[2];
  size_t columns = ins[0].shape()[3];

  CrossMapNormal<Device>(outs[0].data<real>(),
                         outs[1].data<real>(),
                         ins[0].data<real>(),
                         batchSize,
                         maps,
                         rows,
                         columns,
                         attrs.size,
                         attrs.scale,
                         attrs.pow);
  return Error();
}

BEGIN_REGISTER_FUNCTION(CrossMapNormalFwd, forward, CrossMapNormalAttribute)
setDescription(R"DOC(Normalization with across maps Forward functions.

This Function comes from the paper "ImageNet Classification with Deep
Convolutional Neural Networks"

The original formula is

  b_{x,y}^i = \frac{a_{x,y}^i}{ k + \left ( \alpha
                  \sum_{j=max(0, c-N/2)}^{j=min(C, c+N/2)}
                  \left ( a_{x,y}^j \right )^\beta \right )}

Where a is the Input, b is the Output. The {x,y} is the {height, width} of the
image. The i is the channel of the image.

The input and output of this funciton is NCHW format, while the dimension of
input.shape is 4: `batch size`, `channels(feature maps)`, `height(rows)`,
`width(columns)`.

C is the number of feature maps of one images, and N is a hyper-parameter, which
configured when Function is initialized. The sum in the denominator is the sum
of the same position in the neighboring maps.

In the implementation of Function, k is equal to 1, so Function has no argument
for k.
)DOC");

addTensor<INPUT>(/*dim=*/4)->setDescription(
    "Input Image of cross map normalization");
addTensor<OUTPUT>(/*dim=*/4, ASSIGN_TO)
    ->setDescription("Output image of cross map normalization");
addTensor<OUTPUT>(4, ASSIGN_TO)
    ->setDescription(
        "Second output is to simplify the backward calculation. which has same "
        "shape of input. It is the denominator in the equation.");

setShapeInferer([](std::vector<topology::TensorPtr>& ins,
                   std::vector<topology::TensorPtr>& outs) -> Error {
  outs[0]
      ->setShape(ins[0]->shape())
      .setDataType(topology::DataType::DENSE)
      .setSequenceType(ins[0]->sequenceType());
  outs[1]
      ->setShape(ins[0]->shape())
      .setDataType(topology::DataType::DENSE)
      .setSequenceType(ins[0]->sequenceType());
  return Error();
});

setFlopsEstimator<CrossMapNormalAttribute>(
    [](std::vector<topology::TensorPtr>& ins,
       std::vector<topology::TensorPtr>&,
       const CrossMapNormalAttribute& attrs,
       uint64_t* flops) -> Error {
      auto shape = ins[0]->shape();
      auto shapeAccum = std::accumulate(
          shape.begin(), shape.end(), (size_t)1, std::multiplies<size_t>());

      *flops = shapeAccum * (attrs.size * 2 + 3);

      return Error();
    });

END_REGISTER_FUNCTION(CrossMapNormalFwd)

template <DeviceType Device>
static Error backward(const BufferArgs& inputs,
                      const BufferArgs& outputs,
                      const CrossMapNormalAttribute& attrs) {
  if (outputs[0].getArgType() != ADD_TO) {
    // Currently, some algorithm implementations are ASSIGN_TO mode,
    // if need to support the ADD_TO calculation, need to clear the output.
    typename Tensor<real, Device>::Vector tmp(outputs[0].shape().getElements(),
                                              outputs[0].data<real>());
    tmp.zero();
  }

  size_t batchSize = inputs[0].shape()[0];
  size_t maps = inputs[0].shape()[1];
  size_t rows = inputs[0].shape()[2];
  size_t columns = inputs[0].shape()[3];

  CrossMapNormalGrad<Device>(outputs[0].data<real>(),
                             inputs[0].data<real>(),
                             inputs[1].data<real>(),
                             inputs[2].data<real>(),
                             inputs[3].data<real>(),
                             batchSize,
                             maps,
                             rows,
                             columns,
                             attrs.size,
                             attrs.scale,
                             attrs.scale);
  return Error();
}

BEGIN_REGISTER_FUNCTION(CrossMapNormalBwd, backward, CrossMapNormalAttribute)
setDescription(R"DOC(Backward calculation for normalization with across maps.

The equation is
  InputGrad = OutputGrad * Denominator ^{-\beta} +
               \sum_{j=max(0, c-N/2)}^{j=min(C, c+N/2)}
      (OutputGrad * OutputValue * (-2 *\alpha *\beta)/ denominator) * InputValue

The data of inputs/outputs format is the same as the forward interface and is
NCHW.

The upper and lower is the same as forward. The logic of the sum is also the
same as forward.
)DOC");

addTensor<INPUT>(4)->setDescription("InputValue in the equation.");
addTensor<INPUT>(4)->setDescription(
    "OutputValue in the equation. The output[0] of CrossMapNormalFwd "
    "function.");
addTensor<INPUT>(4)->setDescription("OutputGrad in the equation.");
addTensor<INPUT>(4)->setDescription(
    "Denominator in the equation. The output[1] of CrossMapNormalFwd "
    "function.");

// CMR OP supports ADD_TO and ASSIGN_TO
addTensor<OUTPUT>(4)
    ->setDescription("The input grad")
    .supportArgType(ADD_TO, {ADD_TO, ASSIGN_TO});

setShapeInferer([](std::vector<topology::TensorPtr>& ins,
                   std::vector<topology::TensorPtr>& outs) -> Error {
  auto shape = ins[0]->shape();
  for (size_t i = 1; i < ins.size(); ++i) {
    if (shape != ins[i]->shape())
      return Error("Input shape must be same in CMR Projection");
  }
  outs[0]
      ->setShape(shape)
      .setDataType(topology::DataType::DENSE)
      .setSequenceType(ins[0]->sequenceType());

  return Error();
});

setFlopsEstimator<CrossMapNormalAttribute>(
    [](std::vector<topology::TensorPtr>& ins,
       std::vector<topology::TensorPtr>&,
       const CrossMapNormalAttribute& attrs,
       uint64_t* flops) -> Error {
      auto shape = ins[0]->shape();
      *flops = std::accumulate(
                   shape.begin(), shape.end(), 1, std::multiplies<size_t>()) *
               (attrs.size * 4 + 2);
      return Error();
    });

END_REGISTER_FUNCTION(CrossMapNormalBwd)
}  // namespace paddle
