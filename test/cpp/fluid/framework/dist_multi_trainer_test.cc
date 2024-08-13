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

#include "gtest/gtest.h"
#include "paddle/common/flags.h"
#include "paddle/fluid/framework/trainer.h"
#ifdef PADDLE_WITH_GLOO
#include "paddle/fluid/framework/fleet/gloo_wrapper.h"
#endif
#if defined _WIN32 || defined __APPLE__
#else
#define _LINUX
#endif
COMMON_DECLARE_bool(enable_exit_when_partial_worker);

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
  Scope root_scope;
  tmp1->SetScope(&root_scope);
  tmp1->Initialize(t, dataset.get());
  ProgramDesc p;
  tmp1->InitOtherEnv(p);
  tmp1->Finalize();
#endif
}

TEST(DisMultiTrainerTest, testforgpugraph) {
#ifdef _LINUX
  TrainerDesc t;
  t.set_class_name("MultiTrainer");
  t.set_device_worker_name("HogwildWorker");
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
  dataset->SetGpuGraphMode(true);
  dataset->GetMemoryDataSize();
  dataset->SetPassId(2);
  dataset->GetPassID();
  dataset->GetEpochFinish();
#endif
}

TEST(DisMultiTrainerTest, test2) {
#ifdef _LINUX
  FLAGS_enable_exit_when_partial_worker = true;
  std::shared_ptr<MultiTrainer> tmp1 = std::make_shared<MultiTrainer>();
  TrainerDesc t;
  t.set_class_name("MultiTrainer");
  t.set_device_worker_name("HogwildWorker");
  t.set_thread_num(1);
  auto* m = t.mutable_downpour_param()->add_program_config();
  m->set_program_id("123");
  std::string str;
  // str += "name: \"MultiSlotDataFeed\"\nbatch_size: 2\nmulti_slot_desc {\n";
  str +=
      "name: \"SlotRecordInMemoryDataFeed\"\nbatch_size: 2\nmulti_slot_desc "
      "{\n";
  str += "slots {\nname: \"words\"\ntype: \"uint64\"\nis_dense: false\n";
  str += "is_used: true\n}\nslots {\nname: \"label\"\ntype: \"uint64\"\n";
  str += "is_dense: false\nis_used: true\n}\n}\n";
  str += "graph_config {\n";
  str += "gpu_graph_training: true\n}";
  // std::shared_ptr<MultiSlotDataset> dataset =
  //     std::make_shared<MultiSlotDataset>();
  std::shared_ptr<SlotRecordDataset> dataset =
      std::make_shared<SlotRecordDataset>();

  dataset->SetFileList(std::vector<std::string>());
  dataset->SetThreadNum(1);
  dataset->SetTrainerNum(1);
  dataset->SetDataFeedDesc(str);
  dataset->CreateChannel();
  dataset->CreateReaders();
  Scope root_scope;
  tmp1->SetScope(&root_scope);
  tmp1->Initialize(t, dataset.get());
  tmp1->SetDebug(false);
  ProgramDesc p;
  tmp1->InitOtherEnv(p);
  tmp1->Run();
  tmp1->Finalize();
#endif
}

TEST(DisMultiTrainerTest, test3) {
#ifdef _LINUX
  FLAGS_enable_exit_when_partial_worker = true;
  std::shared_ptr<MultiTrainer> tmp1 = std::make_shared<MultiTrainer>();
  TrainerDesc t;
  t.set_class_name("MultiTrainer");
  t.set_device_worker_name("HogwildWorker");
  t.set_thread_num(1);
  auto* m = t.mutable_downpour_param()->add_program_config();
  m->set_program_id("123");
  std::string str;
  // str += "name: \"MultiSlotDataFeed\"\nbatch_size: 2\nmulti_slot_desc {\n";
  str +=
      "name: \"SlotRecordInMemoryDataFeed\"\nbatch_size: 2\nmulti_slot_desc "
      "{\n";
  str += "slots {\nname: \"words\"\ntype: \"uint64\"\nis_dense: false\n";
  str += "is_used: true\n}\nslots {\nname: \"label\"\ntype: \"uint64\"\n";
  str += "is_dense: false\nis_used: true\n}\n}\n";
  str += "graph_config {\n";
  str += "gpu_graph_training: true\n}";
  // std::shared_ptr<MultiSlotDataset> dataset =
  //     std::make_shared<MultiSlotDataset>();
  std::shared_ptr<SlotRecordDataset> dataset =
      std::make_shared<SlotRecordDataset>();

  dataset->SetFileList(std::vector<std::string>());
  dataset->SetThreadNum(1);
  dataset->SetTrainerNum(1);
  dataset->SetDataFeedDesc(str);
  dataset->CreateChannel();
  dataset->SetGpuGraphMode(true);
  dataset->CreateReaders();
  auto readers = dataset->GetReaders();
  readers[0]->SetGpuGraphMode(true);
  Scope root_scope;
  tmp1->SetScope(&root_scope);
  tmp1->Initialize(t, dataset.get());
  tmp1->SetDebug(true);
  ProgramDesc p;
  tmp1->InitOtherEnv(p);
  // tmp1->Run();
  tmp1->Finalize();
#endif
}

}  // namespace framework
}  // namespace paddle
