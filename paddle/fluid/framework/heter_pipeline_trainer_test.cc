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
USE_NO_KERNEL_OP(heter_listen_and_serv);
namespace paddle {
namespace framework {

framework::BlockDesc* AppendSendAndRecvBlock(framework::ProgramDesc* program) {
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
  return block;
}

void GetHeterListenAndServProgram(framework::ProgramDesc* program) {
  auto root_block = program->MutableBlock(0);

  auto* sub_block = AppendSendAndRecvBlock(program);
  std::vector<framework::BlockDesc*> optimize_blocks;
  optimize_blocks.push_back(sub_block);

  std::vector<std::string> message_to_block_id = {"x:1"};
  std::string endpoint = "127.0.0.1:19944";

  framework::OpDesc* op = root_block->AppendOp();
  op->SetType("heter_listen_and_serv");
  op->SetInput("X", {});
  op->SetAttr("message_to_block_id", message_to_block_id);
  op->SetAttr("optimize_blocks", optimize_blocks);
  op->SetAttr("endpoint", endpoint);
  op->SetAttr("fanin", 1);
  op->SetAttr("pserver_id", 0);
}

TEST(HeterPipelineTrainerTest, test1) {
#ifdef _LINUX
  TrainerDesc t, t2;
  // t2
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
  // t2
  t2.set_class_name("HeterPipelineTrainer");
  t2.set_device_worker_name("HeterSectionWorker");
  t2.set_thread_num(1);
  t2.set_trainer_id(1);
  t2.add_trainers(1);
  t2.add_trainers(1);
  auto* heter_section_param2 = t2.mutable_heter_section_param();
  heter_section_param2->set_num_pipeline_stages(2);
  heter_section_param2->set_pipeline_stage(1);
  heter_section_param2->set_num_microbatches(1);

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
  // AppendSendAndRecvBlock(&p);
  GetHeterListenAndServProgram(&p);
  auto* section_config = heter_section_param->mutable_section_config();
  proto::ProgramDesc* pd = new proto::ProgramDesc(*(p.Proto()));
  section_config->set_allocated_program_desc(pd);

  ProgramDesc p2;
  // construct program
  // AppendSendAndRecvBlock(&p2);
  GetHeterListenAndServProgram(&p2);
  auto* section_config2 = heter_section_param2->mutable_section_config();
  proto::ProgramDesc* pd2 = new proto::ProgramDesc(*(p2.Proto()));
  section_config2->set_allocated_program_desc(pd2);

  Scope root_scope, root_scope2;
  paddle::platform::CPUPlace place;

  // tmp1
  std::shared_ptr<TrainerBase> tmp1;
  tmp1 = TrainerFactory::CreateTrainer(t.class_name());
  tmp1->SetScope(&root_scope);
  tmp1->Initialize(t, dataset.get());
  tmp1->InitTrainerEnv(p, place);
  tmp1->InitOtherEnv(p);
  tmp1->GetWorkerScope(0);
  tmp1->ResetDataset(dataset.get());
  tmp1->Finalize();

  // tmp2
  std::shared_ptr<TrainerBase> tmp2;
  tmp2 = TrainerFactory::CreateTrainer(t2.class_name());
  tmp2->SetScope(&root_scope2);
  tmp2->Initialize(t2, dataset.get());
  tmp2->InitTrainerEnv(p2, place);
  tmp2->InitOtherEnv(p2);
  tmp2->GetWorkerScope(0);
  tmp2->ResetDataset(dataset.get());
  tmp2->Finalize();

#endif
}
}  // namespace framework
}  // namespace paddle
#endif
