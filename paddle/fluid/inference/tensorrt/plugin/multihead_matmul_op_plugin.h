// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <algorithm>
#include <string>
#include <vector>
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

class MultiheadMatmulPlugin : public PluginTensorRT {
  bool transpose_q_;
  bool transpose_k_;
  bool transpose_v_;
  float alpha_;
  int head_number_;
  int seq_len_;
  int size_per_head_;

 protected:
  size_t getSerializationSize() override {
    return getBaseSerializationSize() + SerializedSize(transpose_q_) +
           SerializedSize(transpose_k_) + SerializedSize(transpose_v_) +
           SerializedSize(alpha_) + SerializedSize(head_number_) +
           SerializedSize(seq_len_) + SerializedSize(size_per_head_) +
           SerializedSize(getPluginType());
  }

  // TRT will call this func when we need to serialize the configuration of
  // tensorrt.
  // It should not be called by users.
  void serialize(void *buffer) override {
    SerializeValue(&buffer, getPluginType());
    serializeBase(buffer);
    SerializeValue(&buffer, transpose_q_);
    SerializeValue(&buffer, transpose_k_);
    SerializeValue(&buffer, transpose_v_);
    SerializeValue(&buffer, alpha_);
    SerializeValue(&buffer, head_number_);
    SerializeValue(&buffer, seq_len_);
    SerializeValue(&buffer, size_per_head_);
  }

 public:
  MultiheadMatmulPlugin(const bool transpose_q, const bool transpose_k,
                        const bool transpose_v, const float alpha,
                        const int head_number, const int seq_len,
                        const int size_per_head)
      : transpose_q_(transpose_q),
        transpose_k_(transpose_k),
        transpose_v_(transpose_v),
        alpha_(alpha),
        head_number_(head_number),
        seq_len_(seq_len),
        size_per_head_(size_per_head) {}

  // It was used for tensorrt deserialization.
  // It should not be called by users.
  MultiheadMatmulPlugin(void const *serialData, size_t serialLength) {
    deserializeBase(serialData, serialLength);
    DeserializeValue(&serialData, &serialLength, &transpose_q_);
    DeserializeValue(&serialData, &serialLength, &transpose_k_);
    DeserializeValue(&serialData, &serialLength, &transpose_v_);
    DeserializeValue(&serialData, &serialLength, &alpha_);
    DeserializeValue(&serialData, &serialLength, &head_number_);
    DeserializeValue(&serialData, &serialLength, &seq_len_);
    DeserializeValue(&serialData, &serialLength, &size_per_head_);
  }
  ~MultiheadMatmulPlugin() {}
  int initialize() override;

  MultiheadMatmulPlugin *clone() const override {
    return new MultiheadMatmulPlugin(transpose_q_, transpose_k_, transpose_v_,
                                     alpha_, head_number_, seq_len_,
                                     size_per_head_);
  }

  const char *getPluginType() const override {
    return "multihead_matmul_plugin";
  }
  int getNbOutputs() const override { return 1; }
  nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims *inputs,
                                     int nbInputDims) override;
  int enqueue(int batchSize, const void *const *inputs, void **outputs,
              void *workspace, cudaStream_t stream) override;
};

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
