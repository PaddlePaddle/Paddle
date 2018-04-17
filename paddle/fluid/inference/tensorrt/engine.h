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

#pragma once

#include <NvInfer.h>
#include "paddle/fluid/inference/engine.h"

namespace paddle {

/*
 * TensorRT Engine.
 *
 * There are two alternative ways to use it, one is  to build from a paddle
 * protobuf model, another way is to manully construct the network.
 *
 *  TensorEngine some_engine(max_batch, max_workspace);
 *  some_engine.Build(some_paddle_model);
 *  for (auto &batch : some_batch_datas) {
 *    // for all i
 *    some_engine.SetInput("input_i", &batch[i], sizeof(float) *
 * batch[i].size()); some_engine.Execute(); some_engine.GetOutput("output_j");
 * // for all j
 *  }
 */
class TensorrtEngine : public EngineBase {
 public:
  using data_type = nvinfer1::DataType;
  using dim_type = nvinfer1::DimsCHW;

  // Weight is model parameter.
  class Weight {
   public:
    Weight(data_type dtype, void* value, int num_elem) {
      w_.type = dtype;
      w_.values = value;
      w_.count = num_elem;
    }
    const nvinfer1::Weights& get() { return w_; }

   private:
    nvinfer1::Weights w_;
  };

  TensorrtEngine(int max_batch, int max_workspace, cudaStream_t* stream)
      : max_batch_(max_batch), max_workspace_(max_workspace), stream_(stream) {}

  virtual ~TensorrtEngine();

  virtual void Build(const PbType& paddle_model) override;

  virtual void Execute(int batch_size) override;

  // Initialize the infer network, so that layers can add to this network.
  void InitNetwork();
  // Finished adding layers, freeze this network and creates the executation
  // environment.
  void FreezeNetwork();

  // Add an input and set its namd, data type and dimention.
  nvinfer1::ITensor* DeclInput(const std::string& name, data_type dtype,
                               const dim_type& dim);
  // Set the offset-th output from a layer as the network's output, and set its
  // name.
  void DeclOutput(ILayer* layer, int offset, const std ::string& name);

  // GPU memory address for a tensor with specific name. One can operate on
  // these memory directly for acceleration, for example, output the converted
  // data directly to the buffer to save data copy overhead.
  // NOTE this should be used after calling `FreezeNetwork`.
  void* buffer(const std::string& name);

  // Fill an input from CPU memory with name and size.
  void SetInputFromCPU(const std::string& name, void* data, size_t size);
  // TODO(Superjomn) is this method necessary given that raw_input(xxx) can be
  // accessed directly. Fill an input from GPU memory with name and size.
  void SetInputFromGPU(const std::string& name, void* data, size_t size);
  // Get an output called name.
  void* GetOutput(const std::string& name);

  // helper functions to configure the network
  nvinfer1::INetworkDefinition* network() { return infer_network_; }

 private:
  int max_batch_;
  int max_workspace_;
  cudaStream_t* stream_;

  nvinfer1::IBuilder* infer_builder_{nullptr};
  nvinfer1::INetworkDefinition* infer_network_{nullptr};
  nvinfer1::ICudaEngine* infer_engine_{nullptr};
  nvinfer1::IExecutionContext* infer_context_{nullptr};

  std::vector<void*> buffers_;
  // max data size for the buffers.
  std::unordered_map<std::string /*name*/, size_t /*max size*/> buffer_sizes_;
};  // class TensorrtEngine

// Add an layer__ into engine__ with args ARGS.
// For example:
//   TRT_ENGINE_ADD_LAYER(xxx, FullyConnected, input, dim, weights, bias)
//
// Reference
// https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#charRNN_define_network
//
// will add a fully connected layer into the engine.
// TensorRT has too many layers, so that is not wise to add member functions for
// them, and an macro like this is more extensible when underlying TensorRT
// library add new layer supports.
#define TRT_ENGINE_ADD_LAYER(engine__, layer__, ARGS...) \
  engine__->add##layer__(ARGS)

}  // namespace paddle
