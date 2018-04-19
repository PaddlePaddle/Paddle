/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/fluid/framework/framework.pb.h"

namespace paddle {

/*
 * EngineBase is the base class of all inference engines. An inference engine
 * takes a paddle program as input, and output the result in paddle Tensor
 * format. It can be used to optimize performance of computation subgraphs, for
 * example, break down the original model into subgraphs and execute each
 * subgraph in different engines.
 *
 * For example:
 *   When inference, the resnet50 model can put most of the model into subgraph
 * and run it on a TensorRT engine.
 *
 * There are several engines such as TensorRT and other internal frameworks, so
 * an EngineBase is put forward to give an unified interface for all the
 * different engine implemention.
 */
class EngineBase {
 public:
  // TODO fix it latter
  using PbType = int;  // proto::BlockDesc;

  // Build the model and do some preparation, for example, in TensorRT, run
  // createInferBuilder, buildCudaEngine.
  virtual void Build(const PbType& paddle_model) = 0;

  // Execute the engine, that will run the inference network.
  virtual void Execute(int batch_size) = 0;

  virtual ~EngineBase() {}

};  // class EngineBase

}  // namespace paddle
