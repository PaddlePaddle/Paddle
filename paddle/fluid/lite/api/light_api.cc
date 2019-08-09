// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/lite/api/light_api.h"

namespace paddle {
namespace lite {

void LightPredictor::Build(const std::string& model_dir) {
  framework::proto::ProgramDesc desc;
  LoadModel(model_dir, scope_.get(), &desc);
  BuildRuntimeProgram(desc);
}

Tensor* LightPredictor::GetInput(size_t offset) {
  auto* _feed_list = program_->exec_scope()->FindVar("feed");
  CHECK(_feed_list) << "no feed variable in exec_scope";
  auto* feed_list = _feed_list->GetMutable<std::vector<Tensor>>();
  if (offset >= feed_list->size()) {
    feed_list->resize(offset + 1);
  }
  return &feed_list->at(offset);
}

const Tensor* LightPredictor::GetOutput(size_t offset) {
  auto* _fetch_list = program_->exec_scope()->FindVar("fetch");
  CHECK(_fetch_list) << "no fatch variable in exec_scope";
  auto& fetch_list = *_fetch_list->GetMutable<std::vector<lite::Tensor>>();
  CHECK_LT(offset, fetch_list.size()) << "offset " << offset << " overflow";
  return &fetch_list.at(offset);
}

void LightPredictor::BuildRuntimeProgram(
    const framework::proto::ProgramDesc& prog) {
  std::vector<Instruction> insts;
  // 1. Create op first
  Program program(prog, scope_, {});

  // 2. Create Instructs

  // Create the kernels of the target places, and filter out the specific
  // kernel with the target alias.
  for (auto& op : program.ops()) {
    auto kernel_type = op->op_info()->GetAttr<std::string>(kKernelTypeAttr);
    std::string op_type, alias;
    Place place;
    KernelBase::ParseKernelType(kernel_type, &op_type, &alias, &place);
    auto kernels = op->CreateKernels({place});
    // filter out a kernel
    auto it = std::find_if(
        kernels.begin(), kernels.end(),
        [&](std::unique_ptr<KernelBase>& it) { return it->alias() == alias; });
    CHECK(it != kernels.end());
    (*it)->SetContext(ContextScheduler::Global().NewContext((*it)->target()));
    insts.emplace_back(op, std::move(*it));
  }
  program_.reset(new RuntimeProgram(std::move(insts)));
  CHECK(program.exec_scope());
  program_->set_exec_scope(program.exec_scope());
}

LightPredictor::LightPredictor(const std::string& model_dir) {
  scope_ = std::make_shared<Scope>();
  Build(model_dir);
}

}  // namespace lite
}  // namespace paddle
