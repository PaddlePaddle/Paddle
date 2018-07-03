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

#include "ExpandConvLayer.h"
#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"

DEFINE_bool(use_nnpack,
            false,
            "Whether to use nnpack for convolution calculation.");

namespace paddle {

/*
 * The calculation of the exconvt(convolution transpose (deconv) operation)
 * is a swap of forward and backward of the calculation of exconv.
 * */
REGISTER_LAYER(exconv, ExpandConvLayer);
REGISTER_LAYER(exconvt, ExpandConvLayer);

inline bool isDepthwiseConv(int channels, int groups) {
  return channels == groups;
}

bool ExpandConvLayer::init(const LayerMap &layerMap,
                           const ParameterMap &parameterMap) {
  /* Initialize the basic convolutional parent class */
  ConvBaseLayer::init(layerMap, parameterMap);

  int index = 0;
  for (auto &inputConfig : config_.inputs()) {
    const ConvConfig &conf = inputConfig.conv_conf();
    /* Consistent caffe mode for multiple input */
    caffeMode_ = conf.caffe_mode();

    // create a new weight
    size_t height, width;
    height = filterPixels_[index] * filterChannels_[index];
    width = (!isDeconv_) ? numFilters_ : channels_[index];
    CHECK_EQ(parameters_[index]->getSize(), width * height);
    Weight *w = new Weight(height, width, parameters_[index]);
    weights_.emplace_back(w);
    index++;
  }

  if (biasParameter_.get()) {
    if (sharedBiases_) {
      CHECK_EQ((size_t)numFilters_, biasParameter_->getSize());
      biases_ = std::unique_ptr<Weight>(
          new Weight(1, numFilters_, biasParameter_, 0));
    } else {
      biases_ =
          std::unique_ptr<Weight>(new Weight(1, getSize(), biasParameter_, 0));
    }
  }

  getOutputSize();

  size_t numInputs = config_.inputs_size();
  inputShape_.resize(numInputs);
  filterShape_.resize(numInputs);
  outputShape_.resize(numInputs);

  std::string convType;
  std::string convGradInputType;
  std::string convGradFilterType;

  for (int i = 0; i < config_.inputs_size(); i++) {
    std::vector<size_t> paddings = {(size_t)paddingY_[i], (size_t)padding_[i]};
    std::vector<size_t> strides = {(size_t)strideY_[i], (size_t)stride_[i]};
    std::vector<size_t> dilations = {(size_t)dilationY_[i],
                                     (size_t)dilation_[i]};

    bool useDilation = ((size_t)dilationY_[i] > 1 || (size_t)dilation_[i] > 1);

    // Convolution Layer uses the GemmConv function by default.
    convType = "GemmConv";
    convGradInputType = "GemmConvGradInput";
    convGradFilterType = "GemmConvGradFilter";

    // If depth wise convolution and useGpu == true
    if (useGpu_ && isDepthwiseConv(channels_[i], groups_[i]) && !isDeconv_) {
      convType = "DepthwiseConv";
      convGradInputType = "DepthwiseConvGradInput";
      convGradFilterType = "DepthwiseConvGradFilter";
    }

    // If depth wise convolution and useGpu == false and ARM-NEON
    if (!useGpu_ && isDepthwiseConv(channels_[i], groups_[i]) && !isDeconv_) {
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
      if ((filterSize_[i] == filterSizeY_[i]) &&
          (filterSize_[i] == 3 || filterSize_[i] == 4) &&
          (stride_[i] == strideY_[i]) && (stride_[i] == 1 || stride_[i] == 2) &&
          !useDilation) {
        convType = "NeonDepthwiseConv";
      }
#endif
    }

    if (FLAGS_use_nnpack && !isDeconv_ && !useDilation) {
      createFunction(forward_,
                     "NNPACKConv",
                     FuncConfig()
                         .set("paddings", paddings)
                         .set("strides", strides)
                         .set("groups", (size_t)groups_[i])
                         .set("algo", std::string("auto")));
    } else {
      createFunction(forward_,
                     !isDeconv_ ? convType : convGradInputType,
                     FuncConfig()
                         .set("paddings", paddings)
                         .set("strides", strides)
                         .set("dilations", dilations)
                         .set("groups", (size_t)groups_[i]));

      createFunction(backward_,
                     !isDeconv_ ? convGradInputType : convType,
                     FuncConfig()
                         .set("paddings", paddings)
                         .set("strides", strides)
                         .set("dilations", dilations)
                         .set("groups", (size_t)groups_[i]));

      createFunction(backward_,
                     convGradFilterType,
                     FuncConfig()
                         .set("paddings", paddings)
                         .set("strides", strides)
                         .set("dilations", dilations)
                         .set("groups", (size_t)groups_[i]));
    }
  }
  return true;
}

size_t ExpandConvLayer::getOutputSize() {
  CHECK_NE(inputLayers_.size(), 0UL);
  size_t layerSize = ConvBaseLayer::calOutputSize();
  return layerSize;
}

// i is the index of input layers
#define BACKWARD_INPUT(i, inputs, outputs) \
  backward_[2 * i]->calc(inputs, outputs)
#define BACKWARD_FILTER(i, inputs, outputs) \
  backward_[2 * i + 1]->calc(inputs, outputs)

void ExpandConvLayer::forward(PassType passType) {
  Layer::forward(passType);

  size_t batchSize = inputLayers_[0]->getOutputValue()->getHeight();
  resetOutput(batchSize, getOutputSize());

  // Calculate the shape of the input, output, and filter.
  for (size_t i = 0; i < inputLayers_.size(); ++i) {
    inputShape_[i] = TensorShape({(size_t)batchSize,
                                  (size_t)channels_[i],
                                  (size_t)imgSizeH_[i],
                                  (size_t)imgSizeW_[i]});
    filterShape_[i] =
        TensorShape({(size_t)groups_[i],
                     !isDeconv_ ? (size_t)numFilters_ / groups_[i]
                                : (size_t)channels_[i] / groups_[i],
                     !isDeconv_ ? (size_t)channels_[i] / groups_[i]
                                : (size_t)numFilters_ / groups_[i],
                     (size_t)filterSizeY_[i],
                     (size_t)filterSize_[i]});
    outputShape_[i] = TensorShape({(size_t)batchSize,
                                   (size_t)numFilters_,
                                   (size_t)outputH_[i],
                                   (size_t)outputW_[i]});
  }

  // Calculate the output value.
  for (size_t i = 0; i < inputLayers_.size(); ++i) {
    BufferArgs inputs;
    BufferArgs outputs;
    inputs.addArg(*getInputValue(i), inputShape_[i]);
    inputs.addArg(*weights_[i]->getW(), filterShape_[i]);
    outputs.addArg(*getOutputValue(),
                   outputShape_[i],
                   !isDeconv_ && i == 0 ? ASSIGN_TO : ADD_TO);

    forward_[i]->calc(inputs, outputs);
  }

  /* add the bias-vector */
  if (biases_.get()) {
    output_.value->addBias(*biases_->getW(), 1.0, sharedBiases_);
  }

  /* activation */
  forwardActivation();
}

void ExpandConvLayer::backward(const UpdateCallback &callback) {
  backwardActivation();

  MatrixPtr outGrad = getOutputGrad();
  if (biases_ && biases_->getWGrad()) {
    biases_->getWGrad()->collectBias(*getOutputGrad(), 1, sharedBiases_);
    /* Increasing the number of gradient */
    biases_->getParameterPtr()->incUpdate(callback);
  }

  // Calculate the input grad and filter grad.
  for (size_t i = 0; i < inputLayers_.size(); ++i) {
    if (getInputGrad(i)) {
      BufferArgs inputs;
      BufferArgs outputs;
      inputs.addArg(*getOutputGrad(), outputShape_[i]);
      inputs.addArg(*weights_[i]->getW(), filterShape_[i]);
      outputs.addArg(*getInputGrad(i), inputShape_[i], ADD_TO);
      BACKWARD_INPUT(i, inputs, outputs);
    }

    if (weights_[i]->getWGrad()) {
      BufferArgs inputs;
      BufferArgs outputs;
      if (!isDeconv_) {
        inputs.addArg(*getOutputGrad(), outputShape_[i]);
        inputs.addArg(*getInputValue(i), inputShape_[i]);
      } else {
        inputs.addArg(*getInputValue(i), inputShape_[i]);
        inputs.addArg(*getOutputGrad(), outputShape_[i]);
      }
      outputs.addArg(*weights_[i]->getWGrad(), filterShape_[i], ADD_TO);
      BACKWARD_FILTER(i, inputs, outputs);

      /* Increasing the number of gradient */
      weights_[i]->getParameterPtr()->incUpdate(callback);
    }
  }
}

}  // namespace paddle
