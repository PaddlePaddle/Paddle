// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/operators/data/data_reader_op.h"
#include "paddle/fluid/operators/data/nvjpeg_decoder.h"
#include "paddle/fluid/operators/data/map_runner.h"
#include "paddle/fluid/operators/data/pipeline.h"


namespace paddle {
namespace operators {
namespace data {

extern NvjpegDecoderThreadPool* decode_pool;

void ShutDownAllDataLoaders() {
  LOG(ERROR) << "ShutDownAllDataLoaders enter";
  // step 1: shutdown reader
  ReaderManager::Instance()->ShutDown();
  LOG(ERROR) << "ShutDownAllDataLoaders reader_wrapper shutdown finish";
  
  // step 2: shutdown decoder
  if (decode_pool) decode_pool->ShutDown();
  LOG(ERROR) << "ShutDownAllDataLoaders decode_pool shutdown finish";

  // step 3: shutdown MapRunner
  MapRunnerManager::Instance()->ShutDown();
  LOG(ERROR) << "ShutDownAllDataLoaders MapRunner shutdown finish";
  
  // step 3: shutdown Pipeline
  PipelineManager::Instance()->ShutDown();
  LOG(ERROR) << "ShutDownAllDataLoaders Pipeline shutdown finish";
}

void ShutDownReadersAndDecoders(const int64_t program_id) {
  LOG(ERROR) << "ShutDownReadersAndDecoders enter, program_id: " << program_id;
  // step 1: shutdown reader
  ReaderManager::Instance()->ShutDownReader(program_id);

  // step 2: shutdown decoder
  DecoderThreadPoolManager::Instance()->ShutDownDecoder(program_id);
  LOG(ERROR) << "ShutDownReadersAndDecoders finish";
}

void ShutDownMaps(const std::vector<int64_t> program_ids) {
  LOG(ERROR) << "ShutDownMaps enter, maps size: " << program_ids.size();
  for (auto& program_id : program_ids) {
    MapRunnerManager::Instance()->ShutDownMapRunner(program_id);
  }
  LOG(ERROR) << "ShutDownMaps finish";
}

void ShutDownPipeline(const int64_t program_id) {
  LOG(ERROR) << "ShutDownPipeline program_id " << program_id << " enter";
  PipelineManager::Instance()->ShutDownPipeline(program_id);
  LOG(ERROR) << "ShutDownPipeline program_id " << program_id << " finish";
}

}  // namespace data
}  // namespace operators
}  // namespace paddle
