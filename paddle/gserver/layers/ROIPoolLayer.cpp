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

#include "ROIPoolLayer.h"
#include <cfloat>

namespace paddle {

REGISTER_LAYER(roi_pool, ROIPoolLayer);

bool ROIPoolLayer::init(const LayerMap& layerMap,
                        const ParameterMap& parameterMap) {
  Layer::init(layerMap, parameterMap);

  const ROIPoolConfig& layerConf = config_.inputs(0).roi_pool_conf();
  pooledWidth_ = layerConf.pooled_width();
  pooledHeight_ = layerConf.pooled_height();
  spatialScale_ = layerConf.spatial_scale();

  return true;
}

void ROIPoolLayer::forward(PassType passType) {
  Layer::forward(passType);

  const ROIPoolConfig& layerConf = config_.inputs(0).roi_pool_conf();
  height_ = getInput(0).getFrameHeight();
  if (!height_) height_ = layerConf.height();
  width_ = getInput(0).getFrameWidth();
  if (!width_) width_ = layerConf.width();
  channels_ = getInputValue(0)->getWidth() / width_ / height_;

  size_t batchSize = getInput(0).getBatchSize();
  size_t numROIs = getInput(1).getBatchSize();

  MatrixPtr dataValue = getInputValue(0);
  MatrixPtr roiValue = getInputValue(1);
  resetOutput(numROIs, channels_ * pooledHeight_ * pooledWidth_);
  MatrixPtr outputValue = getOutputValue();

  if (useGpu_) {  // TODO(guosheng): implement on GPU later
    MatrixPtr dataCpuBuffer;
    Matrix::resizeOrCreate(dataCpuBuffer,
                           dataValue->getHeight(),
                           dataValue->getWidth(),
                           false,
                           false);
    MatrixPtr roiCpuBuffer;
    Matrix::resizeOrCreate(roiCpuBuffer,
                           roiValue->getHeight(),
                           roiValue->getWidth(),
                           false,
                           false);
    dataCpuBuffer->copyFrom(*dataValue);
    roiCpuBuffer->copyFrom(*roiValue);
    dataValue = dataCpuBuffer;
    roiValue = roiCpuBuffer;
    MatrixPtr outputCpuBuffer;
    Matrix::resizeOrCreate(outputCpuBuffer,
                           outputValue->getHeight(),
                           outputValue->getWidth(),
                           false,
                           false);
    outputCpuBuffer->copyFrom(*outputValue);
    outputValue = outputCpuBuffer;
  }

  real* bottomData = dataValue->getData();
  size_t batchOffset = dataValue->getWidth();
  size_t channelOffset = height_ * width_;
  real* bottomROIs = roiValue->getData();
  size_t roiOffset = roiValue->getWidth();
  size_t poolChannelOffset = pooledHeight_ * pooledWidth_;

  real* outputData = outputValue->getData();
  real* argmaxData = nullptr;
  if (passType != PASS_TEST) {
    Matrix::resizeOrCreate(maxIdxs_,
                           numROIs,
                           channels_ * pooledHeight_ * pooledWidth_,
                           false,
                           false);
    argmaxData = maxIdxs_->getData();
  }

  for (size_t n = 0; n < numROIs; ++n) {
    // the first five elememts of each RoI should be:
    // batch_idx, roi_x_start, roi_y_start, roi_x_end, roi_y_end
    size_t roiBatchIdx = bottomROIs[0];
    size_t roiStartW = round(bottomROIs[1] * spatialScale_);
    size_t roiStartH = round(bottomROIs[2] * spatialScale_);
    size_t roiEndW = round(bottomROIs[3] * spatialScale_);
    size_t roiEndH = round(bottomROIs[4] * spatialScale_);
    CHECK_GE(roiBatchIdx, 0UL);
    CHECK_LT(roiBatchIdx, batchSize);
    size_t roiHeight =
        std::max(roiEndH - roiStartH + 1, static_cast<size_t>(1));
    size_t roiWidth = std::max(roiEndW - roiStartW + 1, static_cast<size_t>(1));
    real binSizeH =
        static_cast<real>(roiHeight) / static_cast<real>(pooledHeight_);
    real binSizeW =
        static_cast<real>(roiWidth) / static_cast<real>(pooledWidth_);
    real* batchData = bottomData + batchOffset * roiBatchIdx;
    for (size_t c = 0; c < channels_; ++c) {
      for (size_t ph = 0; ph < pooledHeight_; ++ph) {
        for (size_t pw = 0; pw < pooledWidth_; ++pw) {
          size_t hstart = static_cast<size_t>(std::floor(ph * binSizeH));
          size_t wstart = static_cast<size_t>(std::floor(pw * binSizeW));
          size_t hend = static_cast<size_t>(std::ceil((ph + 1) * binSizeH));
          size_t wend = static_cast<size_t>(std::ceil((pw + 1) * binSizeW));
          hstart = std::min(
              std::max(hstart + roiStartH, static_cast<size_t>(0)), height_);
          wstart = std::min(
              std::max(wstart + roiStartW, static_cast<size_t>(0)), width_);
          hend = std::min(std::max(hend + roiStartH, static_cast<size_t>(0)),
                          height_);
          wend = std::min(std::max(wend + roiStartW, static_cast<size_t>(0)),
                          width_);

          bool isEmpty = (hend <= hstart) || (wend <= wstart);
          size_t poolIndex = ph * pooledWidth_ + pw;
          outputData[poolIndex] = isEmpty ? 0 : -FLT_MAX;
          if (argmaxData) {
            argmaxData[poolIndex] = -1;
          }

          for (size_t h = hstart; h < hend; ++h) {
            for (size_t w = wstart; w < wend; ++w) {
              size_t index = h * width_ + w;
              if (batchData[index] > outputData[poolIndex]) {
                outputData[poolIndex] = batchData[index];
                if (argmaxData) {
                  argmaxData[poolIndex] = index;
                }
              }
            }
          }
        }
      }
      batchData += channelOffset;
      outputData += poolChannelOffset;
      if (argmaxData) {
        argmaxData += poolChannelOffset;
      }
    }
    bottomROIs += roiOffset;
  }
  if (useGpu_) {
    getOutputValue()->copyFrom(*outputValue);
  }
}

void ROIPoolLayer::backward(const UpdateCallback& callback) {
  MatrixPtr inGradValue = getInputGrad(0);
  MatrixPtr outGradValue = getOutputGrad();
  MatrixPtr roiValue = getInputValue(1);

  if (useGpu_) {
    MatrixPtr inGradCpuBuffer;
    Matrix::resizeOrCreate(inGradCpuBuffer,
                           inGradValue->getHeight(),
                           inGradValue->getWidth(),
                           false,
                           false);
    MatrixPtr outGradCpuBuffer;
    Matrix::resizeOrCreate(outGradCpuBuffer,
                           outGradValue->getHeight(),
                           outGradValue->getWidth(),
                           false,
                           false);
    MatrixPtr roiCpuBuffer;
    Matrix::resizeOrCreate(roiCpuBuffer,
                           roiValue->getHeight(),
                           roiValue->getWidth(),
                           false,
                           false);
    inGradCpuBuffer->copyFrom(*inGradValue);
    outGradCpuBuffer->copyFrom(*outGradValue);
    roiCpuBuffer->copyFrom(*roiValue);
    inGradValue = inGradCpuBuffer;
    outGradValue = outGradCpuBuffer;
    roiValue = roiCpuBuffer;
  }

  real* bottomROIs = roiValue->getData();
  size_t numROIs = getInput(1).getBatchSize();
  size_t roiOffset = getInputValue(1)->getWidth();

  real* inDiffData = inGradValue->getData();
  size_t batchOffset = getInputValue(0)->getWidth();
  size_t channelOffset = height_ * width_;

  real* outDiffData = outGradValue->getData();
  size_t poolChannelOffset = pooledHeight_ * pooledWidth_;
  real* argmaxData = maxIdxs_->getData();

  for (size_t n = 0; n < numROIs; ++n) {
    size_t roiBatchIdx = bottomROIs[0];
    real* batchDiffData = inDiffData + batchOffset * roiBatchIdx;
    for (size_t c = 0; c < channels_; ++c) {
      for (size_t ph = 0; ph < pooledHeight_; ++ph) {
        for (size_t pw = 0; pw < pooledWidth_; ++pw) {
          size_t poolIndex = ph * pooledWidth_ + pw;
          if (argmaxData[poolIndex] > 0) {
            size_t index = static_cast<size_t>(argmaxData[poolIndex]);
            batchDiffData[index] += outDiffData[poolIndex];
          }
        }
      }
      batchDiffData += channelOffset;
      outDiffData += poolChannelOffset;
      argmaxData += poolChannelOffset;
    }
    bottomROIs += roiOffset;
  }

  if (useGpu_) {
    getInputGrad(0)->copyFrom(*inGradValue);
  }
}

}  // namespace paddle
