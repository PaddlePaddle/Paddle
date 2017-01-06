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

#include "PadLayer.h"
#include "paddle/utils/Stat.h"

namespace paddle {

REGISTER_LAYER(pad, PadLayer);

bool PadLayer::init(const LayerMap& layerMap,
                    const ParameterMap& parameterMap) {
  /* Initialize the basic parent class */
  Layer::init(layerMap, parameterMap);

  auto& pad_conf = config_.inputs(0).pad_conf();
  auto& img_conf = pad_conf.image_conf();
  CHECK_EQ(config_.inputs_size(), 1);
  inDims_.push_back(0);
  inDims_.push_back(img_conf.channels());
  inDims_.push_back(img_conf.has_img_size_y() ? img_conf.img_size_y()
                                              : img_conf.img_size());
  inDims_.push_back(img_conf.img_size());

  CHECK_EQ(2UL, pad_conf.pad_c_size());
  CHECK_EQ(2UL, pad_conf.pad_h_size());
  CHECK_EQ(2UL, pad_conf.pad_w_size());
  padc_.push_back(pad_conf.pad_c(0));
  padc_.push_back(pad_conf.pad_c(1));
  padh_.push_back(pad_conf.pad_h(0));
  padh_.push_back(pad_conf.pad_h(1));
  padw_.push_back(pad_conf.pad_w(0));
  padw_.push_back(pad_conf.pad_w(1));

  outDims_.resize(4);
  setOutDims(0);

  createFunction(forward_,
                 "Pad",
                 FuncConfig()
                     .set("padc0", padc_[0])
                     .set("padc1", padc_[1])
                     .set("padh0", padh_[0])
                     .set("padh1", padh_[1])
                     .set("padw0", padw_[0])
                     .set("padw1", padw_[1]));
  createFunction(backward_,
                 "PadGrad",
                 FuncConfig()
                     .set("padc0", padc_[0])
                     .set("padc1", padc_[1])
                     .set("padh0", padh_[0])
                     .set("padh1", padh_[1])
                     .set("padw0", padw_[0])
                     .set("padw1", padw_[1]));

  return true;
}

void PadLayer::setOutDims(int batchSize) {
  outDims_[0] = batchSize;
  outDims_[1] = inDims_[1] + padc_[0] + padc_[1];
  outDims_[2] = inDims_[2] + padh_[0] + padh_[1];
  outDims_[3] = inDims_[3] + padw_[0] + padw_[1];
}

void PadLayer::setTensorDim(int batchSize) {
  CHECK_EQ(inputLayers_.size(), 1UL);
  inDims_[0] = batchSize;
  int h = inputLayers_[0]->getOutput().getFrameHeight();
  if (h != 0) inDims_[2];
  int w = inputLayers_[0]->getOutput().getFrameWidth();
  if (w != 0) inDims_[3];
  setOutDims(batchSize);
}

void PadLayer::forward(PassType passType) {
  Layer::forward(passType);
  MatrixPtr input = inputLayers_[0]->getOutputValue();
  size_t batchSize = input->getHeight();
  setTensorDim(batchSize);
  int size = outDims_[1] * outDims_[2] * outDims_[3];
  resetOutput(batchSize, size);
  MatrixPtr outV = getOutputValue();
  REGISTER_TIMER_INFO("PadForward", getName().c_str());
  forward_[0]->calc({Tensor(input->getData(), inDims_)},
                    {Tensor(outV->getData(), outDims_)},
                    {});
}

void PadLayer::backward(const UpdateCallback& callback) {
  (void)callback;

  MatrixPtr preGrad = inputLayers_[0]->getOutputGrad();
  if (NULL == preGrad) {
    return;
  }
  MatrixPtr outGrad = getOutputGrad();
  REGISTER_TIMER_INFO("PadBackward", getName().c_str());
  backward_[0]->calc({Tensor(outGrad->getData(), outDims_)},
                     {},
                     {Tensor(preGrad->getData(), inDims_)});
}
}  // namespace paddle
