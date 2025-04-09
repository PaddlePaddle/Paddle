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

#include "paddle/fluid/framework/ir/node.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/inference/tensorrt/engine.h"

namespace paddle {
namespace framework {
class OpDesc;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace inference {
namespace tensorrt {

/*
 * Single Op teller definition.
 * One can override this and define a more complex tell logic, considering more
 * issues such as op_desc.
 */
struct Teller {
  virtual bool operator()(const framework::OpDesc& desc,
                          bool use_no_calib_int8 = false,
                          bool with_dynamic_shape = false,
                          bool forbid_dynamic_op_enter_into_trt = false,
                          bool use_explicit_quantization = false) = 0;

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

enum class OpConverterType {
  Default = 0,
  GenericPluginCreator,
  CustomPluginCreater,  // typos: disable-line
  CustomGenericPluginCreator
};
/*
 * class OpTeller helps to tell whether a fluid
 * operator can be transformed to a TensorRT layer
 * and use which kind of OpConverter
 */
class OpTeller {
 public:
  static OpTeller& Global() {
    static std::unique_ptr<OpTeller> x(new OpTeller);
    return *x;
  }

  bool Tell(const framework::ir::Node* node,
            bool use_no_calib_int8 = false,
            bool with_dynamic_shape = false,
            bool forbid_dynamic_op_enter_into_trt = false,
            bool use_explicit_quantization = false);

  std::unique_ptr<Teller>& GetDefaultTeller() { return tellers_.at(0); }

  std::unique_ptr<Teller>& GetGenericPluginTeller() { return tellers_.at(1); }

  std::unique_ptr<Teller>& GetCustomPluginTeller() { return tellers_.at(2); }

  std::unique_ptr<Teller>& GetCustomGenericPluginTeller() {
    return tellers_.at(3);
  }

  void SetOpConverterType(framework::OpDesc* op_desc, OpConverterType type) {
    op_desc->SetAttr("converter_type", static_cast<int>(type));
  }

 private:
  OpTeller();

 private:
  std::vector<std::unique_ptr<Teller>> tellers_;
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
