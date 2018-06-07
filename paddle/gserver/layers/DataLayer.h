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

#pragma once

#include <memory>

#include "Layer.h"

namespace paddle {
/**
 * This layer just copy data to output, and has no backward propagation.
 *
 * The config file api is data_layer.
 */
class DataLayer : public Layer {
 public:
  explicit DataLayer(const LayerConfig& config) : Layer(config) {}

  virtual void setData(const Argument& data) { data_ = data; }

  /**
   * Prefetch sparse matrix/ids only.
   */
  void prefetch() override { output_ = data_; }

  /**
   * Forward propagation. Copy data_ (value, in, grad, ids, cpuSequenceDims,
   * sequenceStartPositions, subSequenceStartPositions, strs) to output_.
   */
  void forward(PassType passType) override {
    Layer::forward(passType);
    copyDataToOutput(output_);
    if (FLAGS_show_layer_stat) {
      showOutputStats();
    }
  }

  /**
   * Data layer's backward propagation do nothing.
   */
  void backward(const UpdateCallback& callback) override { (void)callback; }

  void copyOutputToOtherDevice() override {
    for (size_t i = 0; i != outputOtherDevice_.size(); i++) {
      copyDataToOutput(outputOtherDevice_[i]);
    }
  }

 private:
  void copyDataToOutput(Argument& output);

 protected:
  Argument data_;
};

typedef std::shared_ptr<DataLayer> DataLayerPtr;

}  // namespace paddle
