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

#pragma once
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>
#include "paddle/fluid/framework/op_desc.h"

namespace paddle {
namespace inference {
namespace anakin {

/*
 * Single Op teller definition.
 * One can override this and define a more complex tell logic, considerring more
 * issues such as op_desc.
 */
struct Teller {
  virtual bool operator()(const std::string& op_type,
                          const framework::OpDesc& desc) = 0;

  virtual ~Teller() = default;
};
/*
 * A real example:
 *
 * struct SomeTeller : public Teller {
 * bool operator()(const std::string& op_type,
 *                const framework::OpDesc& desc) override {
 *  return op_type == "fc" && desc.Inputs().size() == 2;
 * }
 *};
 */

/*
 * class OpTeller helps to tell whether a fluid
 * operator can be transformed to a TensorRT layer.
 */
class OpTeller {
 public:
  static OpTeller& Global() {
    static std::unique_ptr<OpTeller> x(new OpTeller);
    return *x;
  }

  bool Tell(const std::string& op_type, const framework::OpDesc& desc);

 private:
  OpTeller();

 private:
  std::vector<std::unique_ptr<Teller>> tellers_;
};

}  // namespace anakin
}  // namespace inference
}  // namespace paddle
