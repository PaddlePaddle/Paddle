//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#if defined(PADDLE_WITH_PSCORE)
#include "gtest/gtest.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/framework/trainer.h"
#include "paddle/fluid/framework/trainer_factory.h"

#if defined _WIN32 || defined __APPLE__
#else
#define _LINUX
#endif

USE_OP(scale);
namespace paddle {
namespace framework {

void AppendSendAndRecvBlock(framework::ProgramDesc* program) {
  auto root_block = program->MutableBlock(0);
  auto* block = program->AppendBlock(*root_block);
  auto* block2 = program->AppendBlock(*root_block);

  framework::OpDesc* op = block->AppendOp();
  op->SetType("scale");
  op->SetInput("X", {"x"});
  op->SetOutput("Out", {"res"});
  op->SetAttr("scale", 0.5f);

  framework::OpDesc* op2 = block2->AppendOp();
  op2->SetType("scale");
  op2->SetInput("X", {"x"});
  op2->SetOutput("Out", {"res"});
  op2->SetAttr("scale", 0.5f);

  auto& out = *root_block->Var("res");
  out.SetType(framework::proto::VarType::LOD_TENSOR);
  out.SetShape({1, 10});
}

TEST(HeterPipelineTrainerTest, test1) {
#ifdef _LINUX
  TrainerDesc t;
  t.set_class_name("HeterPipelineTrainer");
  t.set_device_worker_name("HeterSectionWorker");
  t.set_thread_num(1);
  t.set_trainer_id(0);
  t.add_trainers(1);
  t.add_trainers(1);

  auto* heter_section_param = t.mutable_heter_section_param();
  heter_section_param->set_num_pipeline_stages(2);
  heter_section_param->set_pipeline_stage(0);
  heter_section_param->set_num_microbatches(1);
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

  ProgramDesc p;
  // construct program
  AppendSendAndRecvBlock(&p);
  auto* section_config = heter_section_param->mutable_section_config();
  proto::ProgramDesc* pd = new proto::ProgramDesc(*(p.Proto()));
  section_config->set_allocated_program_desc(pd);
  Scope root_scope;
  std::shared_ptr<TrainerBase> tmp1;
  tmp1 = TrainerFactory::CreateTrainer(t.class_name());
  tmp1->SetScope(&root_scope);
  tmp1->Initialize(t, dataset.get());
  paddle::platform::CPUPlace place;
  tmp1->InitTrainerEnv(p, place);
  tmp1->InitOtherEnv(p);
  tmp1->GetWorkerScope(0);
  tmp1->ResetDataset(dataset.get());
  tmp1->Finalize();

#endif
}
}  // namespace framework
}  // namespace paddle
#endif
