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

#pragma once

#include "CaffeBlob.h"
#include "paddle/gserver/layers/Layer.h"

namespace paddle {
/**
 * The config file api is caffe_layer.
 */

class CaffeLayer : public Layer {
protected:
  ::caffe::Layer<real>* caffeOp_;
  std::vector<::caffe::Blob<real> *> bot_, top_, wei_;
  std::vector<bool> propagateDown_;
  // std::vector<std::vector<int>> wDims_;
  int batchSize_;
  int weightNums_;
  /// The number of input. It is not always equal to weightNums_.
  int inputNums_;

public:
  explicit CaffeLayer(const LayerConfig& config)
      : Layer(config), batchSize_(0) {}
  ~CaffeLayer() {}

  bool init(const LayerMap& layerMap, const ParameterMap& parameterMap);
  void resetParameters();

  void forward(PassType passType);
  void backward(const UpdateCallback& callback = nullptr);

  /**
   * Do layer setup. It will create weights that called blobs in Caffe
   * if the blobs are not set. Then it will reshape the top blobs, namely
   * the output of this layer.
   */
  void caffeLayerSetup(int curBatchSize);
};

}  // namespace paddle
