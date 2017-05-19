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
        "shape of input");

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

/**
 * \brief Backward calculation for normalization with across maps.
 *
 * Function implementation:
 *
 * The implementation of this Function is derived from the
 * CrossMapNormalFunc implementation.
 *
 * InputGrad = OutputGrad * denoms ^ (-beta)
 *    -- upper
 *  + > (OutputGrad * OutputValue * (-2 * alpha * beta) / denoms) * InputValue
 *    -- lower
 *
 * The data of inputs/outputs format is the same as the forward interface
 * and is NCHW.
 *
 * The upper and lower is the same as forward. The logic of the sum
 * is also the same as forward.
 *
 * Function Arguments:
 *
 * \param size_      represent N
 * \param scale_     represent alpha
 * \param pow_       represent beta
 * \param inputs[0]  represent InputValue, inputs[0] of CrossMapNormalFunc
 * \param inputs[1]  represent OutputValue, outputs[0] of CrossMapNormalFunc
 * \param inputs[2]  represent OutputGrad
 * \param inputs[3]  represent denoms, outputs[1] of CrossMapNormalFunc
 *                   This is the intermediate result that is
 *                   preserved in the forward calculation.
 * \param outputs[0] represent InputGrad
 */
template <DeviceType Device>
class CrossMapNormalGradFunc : public FunctionBase {
public:
  void init(const function::Config& config) override {
    // function arguments
    size_ = config.get<size_t>("size");
    scale_ = config.get<real>("scale");
    pow_ = config.get<real>("pow");

    // number of inputs and outputs
    numInputs_ = 4;
    numOutputs_ = 1;
  }

  void calc(const BufferArgs& inputs, const BufferArgs& outputs) override {
    check(inputs, outputs);
    if (outputs[0].getArgType() != ADD_TO) {
      // Currently, some algorithm implementations are ASSIGN_TO mode,
      // if need to support the ADD_TO calculation, need to clear the output.
      typename Tensor<real, Device>::Vector tmp(
          outputs[0].shape().getElements(), outputs[0].data<real>());
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
                               size_,
                               scale_,
                               pow_);
  }

  void check(const BufferArgs& inputs, const BufferArgs& outputs) override {
    CHECK_EQ(numInputs_, inputs.size());
    CHECK_EQ(numOutputs_, outputs.size());

    CHECK_EQ(inputs[0].shape().ndims(), (size_t)4);
    CHECK(inputs[0].shape() == inputs[1].shape());
    CHECK(inputs[0].shape() == inputs[2].shape());
    CHECK(inputs[0].shape() == inputs[3].shape());
    CHECK(inputs[0].shape() == outputs[0].shape());
  }

  // Only need the shape of one input, can calculate the
  // floating-point operation.
  size_t ops(const BufferArgs& inputs, const BufferArgs& outputs) override {
    CHECK_LT((size_t)1, inputs.size());
    size_t batchSize = inputs[0].shape()[0];
    size_t maps = inputs[0].shape()[1];
    size_t rows = inputs[0].shape()[2];
    size_t columns = inputs[0].shape()[3];

    // number of floating-point operations
    // an approximate value
    size_t ops = batchSize * maps * rows * columns * (size_ * 4 + 2);

    return ops;
  }

private:
  size_t size_;
  real scale_;
  real pow_;
};

REGISTER_TYPED_FUNC(CrossMapNormalGrad, CPU, CrossMapNormalGradFunc);
#ifndef PADDLE_ONLY_CPU
REGISTER_TYPED_FUNC(CrossMapNormalGrad, GPU, CrossMapNormalGradFunc);
#endif

}  // namespace paddle
