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
#include <iostream>
#include <sstream>
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"
#include "gtest/gtest.h"
#include "paddle/fluid/framework/trainer.h"

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
  t.set_class_name("DistMultiTrainer");
  t.set_device_worker_name("DownpourWorker");
  t.set_thread_num(1);
  auto* m = t.mutable_downpour_param()->add_program_config();
  m->set_program_id("123");
  std::string str;
  str += "name: \"MultiSlotDataFeed\"\nbatch_size: 2\nmulti_slot_desc {\n";
  str += "slots {\nname: \"words\"\ntype: \"uint64\"\nis_dense: false\n";
  str += "is_used: true\n}\nslots {\nname: \"label\"\ntype: \"uint64\"\n";
  str += "is_dense: false\nis_used: true\n}\n}\n";
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
