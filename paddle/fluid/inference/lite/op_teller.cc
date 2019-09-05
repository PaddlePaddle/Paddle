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

namespace paddle {
namespace inference {
namespace lite {

// Just tell by the op_types.
struct SimpleOpTeller : public Teller {
  SimpleOpTeller() {
    ops_.insert("leaky_relu");
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
    control_ops_.insert("while");
    ops_.insert("leaky_relu");
  }

  bool operator()(const std::string& op_type,
                  const framework::OpDesc& op_desc,
                  const framework::ProgramDesc& prog_desc) override {
    if (control_ops_.count(op_type)) {
      LOG(INFO) << "[ControlOpTeller op_type] = " << op_type;
      return true;
    }
    return false;
  }

 private:
  std::unordered_set<std::string> control_ops_;
  std::unordered_set<std::string> ops_;
};




extern std::vector<std::unique_ptr<Teller>> OpTeller::tellers_;
extern std::once_flag OpTeller::init_flag_;

bool OpTeller::Tell(const std::string& op_type, const framework::OpDesc& desc) {
  if (prog_desc_) {
    for (auto& teller : tellers_) {
      if ((*teller)(op_type, desc, *prog_desc_)) return true;
    }
  } else {
    for (auto& teller : tellers_) {
      if ((*teller)(op_type, desc)) return true;
    }
  }
  return false;
}

OpTeller::OpTeller() {
  std::call_once(init_flag_, [this]() {
    tellers_.emplace_back(new SimpleOpTeller);
  });
}

OpTeller::OpTeller(const framework::ProgramDesc& prog_desc) : OpTeller() {
  prog_desc_ = &prog_desc;
};

}  // namespace lite
}  // namespace inference
}  // namespace paddle
