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

#include "Function.h"
#include "Im2Col.h"

namespace paddle {

/*
 * \brief Converts the image data of four dimensions(NCHW) into
 *        a sequence data of three dimensions(NST) in the forward calculation,
 *        which is reversed in the backward calculation.
 *        Where N is batch size, S is the length of the sequence after each
 *        image is expanded, T is the size of each time step in the sequence.
 *
 * Arguments in forward function:
 * \param inputs[0]  Image data of NCHW format.
 * \param outputs[0] Sequence data of NST format.
 *
 * Arguments in backward function:
 * \param inputs[0]  Sequence data of NST format.
 * \param outputs[0] Image data of NCHW format.
 */
class BlockExpandFunction : public FunctionBase {
 public:
  void init(const FuncConfig& config) override {
    // function arguments
    strides_ = config.get<std::vector<size_t>>("strides");
    paddings_ = config.get<std::vector<size_t>>("paddings");
    blocks_ = config.get<std::vector<size_t>>("blocks");

    // number of inputs and outputs
    numInputs_ = 1;
    numOutputs_ = 1;
  }

  void checkShape(const TensorShape& image, const TensorShape& sequence) const {
    // image shape should be 4-dimensional.
    CHECK_EQ(image.ndims(), (size_t)4);
    // sequence shape should be 3-dimensional.
    CHECK_EQ(sequence.ndims(), (size_t)3);
    // The batchSize of the image needs to be equal to
    // the batchSize of the sequence.
    CHECK_EQ(image[0], sequence[0]);
  }

  // Calculate the shape of colData based on the shape of the image
  // and the shape of the sequence.
  TensorShape getColShape(const TensorShape& image,
                          const TensorShape& sequence) const {
    size_t inputChannels = image[1];
    size_t inputHeight = image[2];
    size_t inputWidth = image[3];
    size_t seqLength = sequence[1];
    size_t stepSize = sequence[2];
    size_t outputHeight =
        1 +
        (inputHeight + 2 * paddingH() - blockH() + strideH() - 1) / strideH();
    size_t outputWidth =
        1 +
        (inputWidth + 2 * paddingW() - blockW() + strideW() - 1) / strideW();
    CHECK_EQ(seqLength, outputHeight * outputWidth);
    CHECK_EQ(stepSize, inputChannels * blockH() * blockW());

    // [outputHeight, outputWidth, inputChannels, filterHeight, filterWidth]
    return TensorShape({outputHeight,
                        outputWidth,
                        inputChannels,
                        (size_t)blockH(),
                        (size_t)blockW()});
  }

 protected:
  std::vector<size_t> strides_;
  std::vector<size_t> paddings_;
  std::vector<size_t> blocks_;

  inline int strideH() const { return strides_[0]; }

  inline int strideW() const { return strides_[1]; }

  inline int paddingH() const { return paddings_[0]; }

  inline int paddingW() const { return paddings_[1]; }

  inline int blockH() const { return blocks_[0]; }

  inline int blockW() const { return blocks_[1]; }
};

template <DeviceType Device>
class BlockExpandForward : public BlockExpandFunction {
 public:
  void init(const FuncConfig& config) override {
    BlockExpandFunction::init(config);
  }

  void check(const BufferArgs& inputs, const BufferArgs& outputs) override {
    const TensorShape& image = inputs[0].shape();
    const TensorShape& sequence = outputs[0].shape();
    checkShape(image, sequence);
  }

  void calc(const BufferArgs& inputs, const BufferArgs& outputs) override {
    CHECK_EQ(numInputs_, inputs.size());
    CHECK_EQ(numOutputs_, outputs.size());
    check(inputs, outputs);
    CHECK_EQ(outputs[0].getArgType(), ASSIGN_TO);
    const TensorShape& image = inputs[0].shape();
    const TensorShape& sequence = outputs[0].shape();

    TensorShape imShape = TensorShape({image[1], image[2], image[3]});
    TensorShape colShape = getColShape(image, sequence);
    size_t batchSize = image[0];

    real* imageData = inputs[0].data<real>();
    real* seqData = outputs[0].data<real>();
    Im2ColFunctor<kOCF, Device, real> im2col;
    for (size_t i = 0; i < batchSize; i++) {
      // The result of im2col is [outputHeight, outputWidth,
      // inputChannels, filterHeight, filterWidth], and it is easy to
      // reshape into [seqLength, stepSize], where seqLength is equal
      // output_height * output_width, stepSize is equal
      // input_channels * filter_height * filter_width
      im2col(imageData,
             imShape,
             seqData,
             colShape,
             strideH(),
             strideW(),
             paddingH(),
             paddingW());
      imageData += imShape.getElements();
      seqData += colShape.getElements();
    }
  }
};

template <DeviceType Device>
class BlockExpandBackward : public BlockExpandFunction {
 public:
  void init(const FuncConfig& config) override {
    BlockExpandFunction::init(config);
  }

  void check(const BufferArgs& inputs, const BufferArgs& outputs) override {
    const TensorShape& image = outputs[0].shape();
    const TensorShape& sequence = inputs[0].shape();
    checkShape(image, sequence);
  }

  void calc(const BufferArgs& inputs, const BufferArgs& outputs) override {
    CHECK_EQ(numInputs_, inputs.size());
    CHECK_EQ(numOutputs_, outputs.size());
    check(inputs, outputs);
    // Since the implementation of Col2ImFunctor is ADD_TO,
    // this function only supports ADD_TO mode.
    CHECK_EQ(outputs[0].getArgType(), ADD_TO);
    const TensorShape& image = outputs[0].shape();
    const TensorShape& sequence = inputs[0].shape();

    TensorShape imShape = TensorShape({image[1], image[2], image[3]});
    TensorShape colShape = getColShape(image, sequence);
    size_t batchSize = image[0];

    real* imageData = outputs[0].data<real>();
    real* seqData = inputs[0].data<real>();
    Col2ImFunctor<kOCF, Device, real> col2im;
    for (size_t i = 0; i < batchSize; i++) {
      col2im(imageData,
             imShape,
             seqData,
             colShape,
             strideH(),
             strideW(),
             paddingH(),
             paddingW());
      imageData += imShape.getElements();
      seqData += colShape.getElements();
    }
  }
};

REGISTER_TYPED_FUNC(BlockExpand, CPU, BlockExpandForward);
REGISTER_TYPED_FUNC(BlockExpandGrad, CPU, BlockExpandBackward);
#ifdef PADDLE_WITH_CUDA
REGISTER_TYPED_FUNC(BlockExpand, GPU, BlockExpandForward);
REGISTER_TYPED_FUNC(BlockExpandGrad, GPU, BlockExpandBackward);
#endif

}  // namespace paddle
