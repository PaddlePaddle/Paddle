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
#include "paddle/fluid/operators/file_label_reader_op.h"
#include "paddle/fluid/operators/data/nvjpeg_decoder.h"
#include "paddle/fluid/operators/data/map_runner.h"
#include "paddle/fluid/operators/data/pipeline.h"


namespace paddle {
namespace operators {

extern FileDataReaderWrapper reader_wrapper;

namespace data {

extern NvjpegDecoderThreadPool* decode_pool;

void ShutDownDataLoader() {
  LOG(ERROR) << "ShutDownDataLoader enter";
  // step 1: shutdown reader
  reader_wrapper.ShutDown();
  LOG(ERROR) << "ShutDownDataLoader reader_wrapper shutdown finish";

  // step 2: shutdown decoder
  if (decode_pool) decode_pool->ShutDown();
  LOG(ERROR) << "ShutDownDataLoader decode_pool shutdown finish";

  // step 3: shutdown MapRunner
  MapRunnerManager::Instance()->ShutDown();
  LOG(ERROR) << "ShutDownDataLoader MapRunner shutdown finish";
}
}  // namespace data

}  // namespace operators
}  // namespace paddle
