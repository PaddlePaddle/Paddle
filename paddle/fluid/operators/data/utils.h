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
#include "paddle/fluid/operators/data/image_decoder.h"
#include "paddle/fluid/operators/data/map_runner.h"
#include "paddle/fluid/operators/data/pipeline.h"


namespace paddle {
namespace operators {
namespace data {

extern ImageDecoderThreadPool* decode_pool;

void ShutDownAllDataLoaders() {
  VLOG(4) << "ShutDownAllDataLoaders enter";
  // step 1: shutdown reader
  ReaderManager::Instance()->ShutDown();
  
  // step 2: shutdown decoder
  if (decode_pool) decode_pool->ShutDown();

  // step 3: shutdown MapRunner
  MapRunnerManager::Instance()->ShutDown();
  
  // step 3: shutdown Pipeline
  PipelineManager::Instance()->ShutDown();
  VLOG(4) << "ShutDownAllDataLoaders Pipeline shutdown finish";
}

void ShutDownReadersAndDecoders(const int64_t program_id) {
  // step 1: shutdown reader
  ReaderManager::Instance()->ShutDownReader(program_id);

  // step 2: shutdown decoder
  ImageDecoderThreadPoolManager::Instance()->ShutDownDecoder(program_id);
}

void ShutDownMaps(const std::vector<int64_t> program_ids) {
  for (auto& program_id : program_ids) {
    MapRunnerManager::Instance()->ShutDownMapRunner(program_id);
  }
}

void ShutDownPipeline(const int64_t program_id) {
  PipelineManager::Instance()->ShutDownPipeline(program_id);
}

void ResetDataLoader(const int64_t reader_id,
                     const std::vector<int64_t> map_ids,
                     const int64_t pipeline_id) {
  VLOG(4) << "ResetDataLoader enter, reader_id: " << reader_id \
          << ", map_ids size: " << map_ids.size() << ", pipeline_id: " \
          << pipeline_id;
  // step 1: reset readers
  ReaderManager::Instance()->ResetReader(reader_id);

  // step 2: reset maps
  for (auto& map_id : map_ids) {
    MapRunnerManager::Instance()->ResetMapRunner(map_id);
  }

  // step3: reset pipeline
  PipelineManager::Instance()->ResetPipeline(pipeline_id);
  VLOG(4) << "ResetDataLoader finish";
}

}  // namespace data
}  // namespace operators
}  // namespace paddle
