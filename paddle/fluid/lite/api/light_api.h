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

/*
 * This file implements a light-weight API which can run on mobile. We limit the
 * dependencies and the runtime computation complexity.
 */
#pragma once

#include <string>
#include <vector>
#include "paddle/fluid/lite/core/program.h"
#include "paddle/fluid/lite/core/types.h"
#include "paddle/fluid/lite/model_parser/model_parser.h"
#include "paddle/fluid/lite/model_parser/pb/op_desc.h"

namespace paddle {
namespace lite {

class CxxPredictor {
 public:
  CxxPredictor() { scope_ = std::make_shared<Scope>(); }

  void Build(const std::string& model_dir) {
    framework::proto::ProgramDesc desc;
    LoadModel(model_dir, scope_.get(), &desc);
    BuildRuntimeProgram(desc);
  }

  void Run() { program_->Run(); }

  // Get offset-th col of feed.
  Tensor* GetInput(size_t offset) {
    auto* _feed_list = program_->exec_scope()->FindVar("feed");
    CHECK(_feed_list) << "no feed variable in exec_scope";
    auto* feed_list = _feed_list->GetMutable<std::vector<Tensor>>();
    if (offset >= feed_list->size()) {
      feed_list->resize(offset + 1);
    }
    return &feed_list->at(offset);
  }

  const Tensor* GetOutput(size_t offset) {
    auto* _fetch_list = program_->exec_scope()->FindVar("fetch");
    CHECK(_fetch_list) << "no fatch variable in exec_scope";
    auto& fetch_list = *_fetch_list->GetMutable<std::vector<lite::Tensor>>();
    CHECK_LT(offset, fetch_list.size()) << "offset " << offset << " overflow";
    return &fetch_list.at(offset);
  }

 private:
  void BuildRuntimeProgram(const framework::proto::ProgramDesc& prog) {
    std::vector<Instruct> insts;
    // 1. Create op first
    Program program(prog, scope_, {});

    // 2. Create Instructs

    // Create the kernels of the target places, and filter out the specific
    // kernel with the target alias.
    for (auto& op : program.ops) {
      lite::pb::OpDesc desc(op->op_info()->desc());
      auto kernel_type = desc.GetAttr(kKernelTypeAttr).get<std::string>();
      std::string op_type, alias;
      Place place;
      KernelBase::ParseKernelType(kernel_type, &op_type, &alias, &place);
      auto kernels = op->CreateKernels({place});
      // filter out a kernel
      auto it = std::find_if(kernels.begin(), kernels.end(),
                             [&](std::unique_ptr<KernelBase>& it) {
                               return it->alias() == alias;
                             });
      CHECK(it != kernels.end());
      insts.emplace_back(op, std::move(*it));
    }
    program_.reset(new RuntimeProgram(std::move(insts)));
    CHECK(program.exec_scope);
    program_->set_exec_scope(program.exec_scope);
  }

 private:
  std::shared_ptr<Scope> scope_;
  std::unique_ptr<RuntimeProgram> program_;
};

}  // namespace lite
}  // namespace paddle
