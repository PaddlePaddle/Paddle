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
#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/program_desc.h"

namespace paddle {
namespace inference {
namespace lite {

// Just tell by the op_types.
struct SimpleOpTeller : public Teller {
  SimpleOpTeller() {
    ops_.insert("leaky_relu");
    ops_.insert("fill_constant");
    ops_.insert("write_to_array");
    ops_.insert("increment");
    ops_.insert("less_than");
    ops_.insert("read_from_array");
    ops_.insert("scale");
    ops_.insert("sum");
  }

  bool operator()(const std::string& op_type,
                  const framework::OpDesc& op_desc) override {
    return ops_.count(op_type);
  }

 private:
  std::unordered_set<std::string> ops_;
};

struct ControlOpTeller : public Teller {
  ControlOpTeller() {
    ops_.insert("while");
  }

  bool operator()(const std::string& op_type,
                  const framework::OpDesc& op_desc) override {
    if (ops_.count(op_type)) {
      SimpleOpTeller supported;
      const int id = op_desc.GetBlockAttrId("sub_block");
      LOG(INFO) << "===== " << op_type << " =====";
      LOG(INFO) << "id: " << id;
      const framework::BlockDesc& block_desc = op_desc.Block()->Program()->Block(id);
      const std::vector<framework::OpDesc *>& ops_sub_block = block_desc.AllOps();
      for (auto* op: ops_sub_block) {
        LOG(INFO) << "block " << id << ", op->Type: " << op->Type();
        if (!supported(op->Type(), *op) && !this->operator()(op->Type(), *op)) {
          return false;
        };
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
  tellers_.emplace_back(new ControlOpTeller);
}

}  // namespace lite
}  // namespace inference
}  // namespace paddle


