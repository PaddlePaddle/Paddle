/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"
#include "CudnnConvLayer.h"

namespace paddle {

REGISTER_LAYER(cudnn_conv, CudnnConvLayer);

bool CudnnConvLayer::init(const LayerMap &layerMap,
                          const ParameterMap &parameterMap) {
  ConvBaseLayer::init(layerMap, parameterMap);
  CHECK(useGpu_) << "CudnnConvLayer only support gpu";

  maxGroups_ = 0;
  for (size_t i = 0; i < inputLayers_.size(); i++) {
    CHECK_EQ(channels_[i] % groups_[i], 0);
    CHECK_EQ(numFilters_ % groups_[i], 0);

    hl_filter_descriptor filter;
    hl_create_filter_descriptor(&filter, channels_[i] / groups_[i],
                                numFilters_ / groups_[i], filterSizeY_[i],
                                filterSize_[i]);
    filterDesc_.push_back(filter);

    hl_tensor_descriptor input;
    hl_create_tensor_descriptor(&input);
    inputDesc_.push_back(input);

    hl_tensor_descriptor output;
    int outputX =
        outputSize(imgSize_[i], filterSize_[i], padding_[i], stride_[i]);
    CHECK_EQ(outputX, outputX_[i]);
    hl_create_tensor_descriptor(&output);
    outputDesc_.push_back(output);

    hl_convolution_descriptor conv;
    hl_create_convolution_descriptor(&conv, input, filter, paddingY_[i],
                                     padding_[i], strideY_[i], stride_[i]);
    convDesc_.push_back(conv);

    weightOffset_.push_back((numFilters_ / groups_[i]) *
                            (channels_[i] / groups_[i]) * filterPixels_[i]);
    inputOffset_.push_back((channels_[i] / groups_[i]) * imgSize_[i] *
                           imgSize_[i]);
    outputOffset_.push_back((numFilters_ / groups_[i]) * outputX_[i] *
                            outputX_[i]);

    // initialize all to default algorithms
    fwdAlgo_.push_back(0);
    bwdFilterAlgo_.push_back(0);
    bwdDataAlgo_.push_back(0);
    fwdLimitBytes_.push_back(0);
    bwdFilterLimitBytes_.push_back(0);
    bwdDataLimitBytes_.push_back(0);

    // cudnn streams per group equal to 1
    if (groups_[i] > maxGroups_) {
      maxGroups_ = groups_[i];
    }
  }

  workSpaceInBytes_ = 0;
  workSpaceData_ = NULL;
  for (int i = 0; i < maxGroups_; ++i) {
    workSpace_.push_back(NULL);
  }

  if (biases_.get() && sharedBiases_) {
    hl_create_tensor_descriptor(&biasDesc_);
    hl_tensor_reshape(biasDesc_, 1, numFilters_ / groups_[0], 1, 1);
    biasOffset_ = numFilters_ / groups_[0];
  }

  isSelectAlgo_ = false;
  return true;
}

void CudnnConvLayer::allocConvWorkSpace(size_t maxWorkSpace) {
  size_t totalWorkSpace = maxWorkSpace * maxGroups_;

  if (totalWorkSpace  > workSpaceInBytes_) {
      if (workSpaceInBytes_ != 0) {
          hl_free_mem_device(workSpaceData_);
      }
      // total amount of storage needed over all groups
      workSpaceData_ = hl_malloc_device(totalWorkSpace);

      // update work space address for each group
      for (int i = 0; i < maxGroups_; ++i) {
            workSpace_[i] = reinterpret_cast<char *>(workSpaceData_)
                                  + i * maxWorkSpace;
      }
      workSpaceInBytes_ = totalWorkSpace;
  }
}

void CudnnConvLayer::reshape(int batchSize) {
  CHECK_NE(inputLayers_.size(), 0UL);
  imageH_ = inputLayers_[0]->getOutput().getFrameHeight();
  imageW_ = inputLayers_[0]->getOutput().getFrameWidth();
  if (imageH_ == 0) imageH_ = imgSize_[0];
  if (imageW_ == 0) imageW_ = imgSize_[0];

  for (size_t i = 1; i < inputLayers_.size(); i++) {
    int imageH = inputLayers_[i]->getOutput().getFrameHeight();
    int imageW = inputLayers_[i]->getOutput().getFrameWidth();
    if (imageH) {
      CHECK_EQ(imageH_, imageH) << "Inputs must have same height.";
    }
    if (imageW) {
      CHECK_EQ(imageW_, imageW) << "Inputs must have same width.";
    }
  }

  outputH_ = outputSize(imageH_, filterSizeY_[0], paddingY_[0], strideY_[0]);
  outputW_ = outputSize(imageW_, filterSize_[0], padding_[0], stride_[0]);
  // check outputH & outputW
  getOutput().setFrameHeight(outputH_);
  getOutput().setFrameWidth(outputW_);

  size_t maxWorkSpace = 0;
  for (size_t i = 0; i < inputLayers_.size(); i++) {
    CHECK_EQ(inputLayers_[i]->getOutput().value->getWidth(),
             (size_t)(channels_[i] * imageH_ * imageW_));

    hl_tensor_reshape(inputDesc_[i], batchSize, channels_[i] / groups_[i],
                      imageH_, imageW_, channels_[i] * imageH_ * imageW_,
                      imageH_ * imageW_, imageW_, 1);

    hl_tensor_reshape(outputDesc_[i], batchSize, numFilters_ / groups_[i],
                      outputH_, outputW_, numFilters_ * outputH_ * outputW_,
                      outputH_ * outputW_, outputW_, 1);

    hl_reset_convolution_descriptor(convDesc_[i], inputDesc_[i],
                                    filterDesc_[i], paddingY_[i],
                                    padding_[i], strideY_[i], stride_[i]);

    inputOffset_[i] = (channels_[i] / groups_[i]) * imageH_ * imageW_;
    outputOffset_[i] = (numFilters_ / groups_[i]) * outputH_ * outputW_;

    if (!isSelectAlgo_) {
      hl_conv_workspace(inputDesc_[i], outputDesc_[i], filterDesc_[i],
                        convDesc_[i], &fwdAlgo_[i], &fwdLimitBytes_[i],
                        &bwdDataAlgo_[i], &bwdDataLimitBytes_[i],
                        &bwdFilterAlgo_[i], &bwdFilterLimitBytes_[i]);

      maxWorkSpace = std::max(fwdLimitBytes_[i], bwdDataLimitBytes_[i]);
      maxWorkSpace = std::max(maxWorkSpace, bwdFilterLimitBytes_[i]);
    }
  }

  if (!isSelectAlgo_) {
    allocConvWorkSpace(maxWorkSpace);
  }

  isSelectAlgo_ = true;
}

void CudnnConvLayer::forward(PassType passType) {
  Layer::forward(passType);
  int batchSize = inputLayers_[0]->getOutputValue()->getHeight();
  reshape(batchSize);
  resetOutput(batchSize, outputH_ * outputW_ * numFilters_);

  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    REGISTER_TIMER_INFO("CudnnConvFwTimer", getName().c_str());
    for (int g = 0; g < groups_[i]; ++g) {
      real *inputData = getInputValue(i)->getData() + inputOffset_[i] * g;
      real *wgtData = weights_[i]->getW()->getData() + weightOffset_[i] * g;
      real *outData = getOutputValue()->getData() + outputOffset_[i] * g;
      hl_convolution_forward(inputDesc_[i], inputData, outputDesc_[i],
                             outData, filterDesc_[i], wgtData,
                             convDesc_[i], workSpace_[g],
                             fwdLimitBytes_[i], fwdAlgo_[i]);
    }
  }

  if (biases_) {
    REGISTER_TIMER_INFO("CudnnConvBiasTimer", getName().c_str());
    addBiases();
  }

  forwardActivation();
}

void CudnnConvLayer::addBiases() {
  if (sharedBiases_) {
    for (int g = 0; g < groups_[0]; ++g) {
      real *biasData = biases_->getW()->getData() + biasOffset_ * g;
      real *outData = getOutputValue()->getData() + outputOffset_[0] * g;
      hl_convolution_forward_add_bias(biasDesc_, biasData,
                                      outputDesc_[0], outData);
    }
  } else {
    LOG(FATAL) << "Not supported";
  }
}

void CudnnConvLayer::bpropBiases() {
  if (sharedBiases_) {
    for (int g = 0; g < groups_[0]; ++g) {
      real *biasGrad = biases_->getWGrad()->getData() + biasOffset_ * g;
      real *outGrad = getOutputGrad()->getData() + outputOffset_[0] * g;
      hl_convolution_backward_bias(biasDesc_, biasGrad,
                                   outputDesc_[0], outGrad);
    }
  } else {
    LOG(FATAL) << "Not supported";
  }
}

void CudnnConvLayer::backward(const UpdateCallback &callback) {
  backwardActivation();

  if (biases_ && biases_->getWGrad()) {
    REGISTER_TIMER_INFO("CudnnConvBpBiasTimer", getName().c_str());
    bpropBiases();
    biases_->getParameterPtr()->incUpdate(callback);
  }

  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    REGISTER_TIMER_INFO("CudnnConvBpTimer", getName().c_str());
    for (int g = 0; g < groups_[i]; ++g) {
      real *outGrad = getOutputGrad()->getData() + outputOffset_[i] * g;
      if (weights_[i]->getWGrad()) {
        real *inputData = getInputValue(i)->getData() + inputOffset_[i] * g;
        real *weightGrad =
            weights_[i]->getWGrad()->getData() + weightOffset_[i] * g;
        hl_convolution_backward_filter(
            inputDesc_[i], inputData, outputDesc_[i], outGrad, filterDesc_[i],
            weightGrad, convDesc_[i], workSpace_[g], bwdFilterLimitBytes_[i],
            bwdFilterAlgo_[i]);
      }

      MatrixPtr preGrad = getInputGrad(i);
      if (NULL != preGrad) {
        real *inputGrad = preGrad->getData() + inputOffset_[i] * g;
        real *wgtData = weights_[i]->getW()->getData() + weightOffset_[i] * g;
        hl_convolution_backward_data(
            inputDesc_[i], inputGrad, outputDesc_[i], outGrad, filterDesc_[i],
            wgtData, convDesc_[i], workSpace_[g], bwdDataLimitBytes_[i],
            bwdDataAlgo_[i]);
      }
    }
    weights_[i]->getParameterPtr()->incUpdate(callback);
  }
}

CudnnConvLayer::~CudnnConvLayer() {
  if (biasDesc_) {
    hl_destroy_tensor_descriptor(biasDesc_);
  }

  for (size_t i = 0; i < inputDesc_.size(); i++) {
    hl_destroy_tensor_descriptor(inputDesc_[i]);
    hl_destroy_tensor_descriptor(outputDesc_[i]);
    hl_destroy_filter_descriptor(filterDesc_[i]);
    hl_destroy_convolution_descriptor(convDesc_[i]);
  }
  if (workSpaceInBytes_ != 0) {
    hl_free_mem_device(workSpaceData_);
    workSpaceInBytes_ = 0;
  }
}

}  // namespace paddle
