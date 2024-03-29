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

#include "paddle/fluid/inference/lite/op_teller.h"

#include <map>

#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/inference/lite/engine.h"

namespace paddle {
namespace lite {
std::vector<std::string> GetAllOps();
}
}  // namespace paddle

namespace paddle {
namespace inference {
namespace lite {

// Just tell by the op_types.
struct SimpleOpTeller : public Teller {
  SimpleOpTeller() {
    std::vector<std::string> lite_ops = paddle::lite::GetAllOps();
    auto is_non_inst = [](const std::string& op) -> bool {
      const std::vector<std::string> ops = {"feed", "fetch", "while"};
      return std::find(ops.begin(), ops.end(), op) != ops.end();
    };
    for (const auto& op : lite_ops) {
      if (!is_non_inst(op)) {
        ops_.insert(op);
      }
    }
  }

  bool operator()(const std::string& op_type,
                  const framework::OpDesc& op_desc) override {
    return ops_.count(op_type);
  }

 private:
  std::unordered_set<std::string> ops_{};
};

struct SingleBlockOpTeller : public Teller {
  SingleBlockOpTeller() { ops_.insert("while"); }

  bool operator()(const std::string& op_type,
                  const framework::OpDesc& op_desc) override {
    if (ops_.count(op_type)) {
      SimpleOpTeller supported;
      const int id = op_desc.GetBlockAttrId("sub_block");
      const framework::BlockDesc& block_desc =
          op_desc.Block()->Program()->Block(id);
      const std::vector<framework::OpDesc*>& ops_sub_block =
          block_desc.AllOps();
      for (auto* op : ops_sub_block) {
        if (!supported(op->Type(), *op) && !this->operator()(op->Type(), *op)) {
          return false;
        }
      }
      return true;
    }
    return false;
  }

 private:
  std::unordered_set<std::string> ops_;
};

bool OpTeller::Tell(const std::string& op_type, const framework::OpDesc& desc) {
  for (auto& teller : tellers_) {
    if ((*teller)(op_type, desc)) return true;
  }
  return false;
}

OpTeller::OpTeller() {
  tellers_.emplace_back(new SimpleOpTeller);
  tellers_.emplace_back(new SingleBlockOpTeller);
}

}  // namespace lite
}  // namespace inference
}  // namespace paddle
