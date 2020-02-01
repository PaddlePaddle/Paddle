//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <fstream>
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"
#include "gtest/gtest.h"
#include <iostream>
#include "paddle/fluid/framework/trainer.h"
#include <sstream>

#if defined _WIN32 || defined __APPLE__
#else
#define _LINUX
#endif

namespace paddle {
namespace framework {
TEST(DisMultiTrainerTest, test1) {
#ifdef _LINUX
  std::shared_ptr<DistMultiTrainer> tmp1 = std::make_shared<DistMultiTrainer>();
  TrainerDesc t;
  std::string str;
  str += "name: \"MultiSlotDataFeed\"\n";
  str += "batch_size: 2\n";
  str += "multi_slot_desc {\n";
  str += "    slots {\n";
  str += "         name: \"words\"\n";
  str += "         type: \"uint64\"\n";
  str += "         is_dense: false\n";
  str += "         is_used: true\n";
  str += "     }\n";
  str += "     slots {\n";
  str += "         name: \"label\"\n";
  str += "         type: \"uint64\"\n";
  str += "         is_dense: false\n";
  str += "         is_used: true\n";
  str += "    }\n";
  str += "}\n";
  std::shared_ptr<MultiSlotDataset> dataset =
      std::make_shared<MultiSlotDataset>();
  dataset->SetFileList(std::vector<std::string>());
  dataset->SetThreadNum(1);
  dataset->SetTrainerNum(1);
  dataset->SetDataFeedDesc(str);
  dataset->CreateReaders();
  tmp1->Initialize(t, dataset.get());
#endif
}
}  // namespace framework
}  // namespace paddle
