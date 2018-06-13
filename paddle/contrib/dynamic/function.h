// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>

#include "paddle/contrib/dynamic/tape.h"
#include "paddle/contrib/dynamic/variable.h"
#include "paddle/fluid/framework/type_defs.h"

namespace paddle {
namespace dynamic {

class Function {};

class Fill {
 public:
  Fill(const std::string &initializer, const framework::AttributeMap &attrs)
      : initializer_(initializer), attrs_(attrs) {}

  void operator()(VariableHandle var) {
    get_global_tape().AddOp(initializer_, {}, {{"Out", {var}}}, attrs_);
  }

 private:
  const std::string initializer_;
  const framework::AttributeMap attrs_;
};

class Mean {
 public:
  VariableHandle operator()(VariableHandle var) {
    VariableHandle out(new Variable("mean"));
    get_global_tape().AddOp("mean", {{"X", {var}}}, {{"Out", {out}}}, {});
    return out;
  }
};

class Linear {
 public:
  Linear(int in_dim, int out_dim, const std::string &act)
      : w_(new Variable("LinearWeight")),
        b_(new Variable("LinearBias")),
        act_(act) {
    Tape init_tape;

    std::string initializer = "fill_constant";
    framework::AttributeMap attrs;
    attrs["dtype"] = paddle::framework::proto::VarType::Type::VarType_Type_FP32;
    attrs["shape"] = std::vector<int>{in_dim, out_dim};
    attrs["value"] = 1.0f;
    init_tape.AddOp(initializer, {}, {{"Out", {w_}}}, attrs);

    attrs["dtype"] = paddle::framework::proto::VarType::Type::VarType_Type_FP32;
    attrs["shape"] = std::vector<int>{out_dim};
    attrs["value"] = 1.0f;
    init_tape.AddOp(initializer, {}, {{"Out", {b_}}}, attrs);

    init_tape.Forward();
  }

  VariableHandle operator()(VariableHandle input) {
    VariableHandle pre_bias(new Variable("linear"));
    get_global_tape().AddOp("mul",
                            {{"X", {input}}, {"Y", {w_}}},
                            {{"Out", {pre_bias}}},
                            {{"x_num_col_dims", 1}, {"y_num_col_dims", 1}});
    VariableHandle pre_act(new Variable("linear"));
    get_global_tape().AddOp("elementwise_add",
                            {{"X", {pre_bias}}, {"Y", {b_}}},
                            {{"Out", {pre_act}}},
                            {{"axis", 1}});
    VariableHandle post_act(new Variable("linear"));
    get_global_tape().AddOp(
        act_, {{"X", {pre_act}}}, {{"Out", {post_act}}}, {});
    return post_act;
  }

  std::vector<VariableHandle> Params() { return {w_, b_}; }

 private:
  VariableHandle w_;
  VariableHandle b_;
  std::string act_;
};

class SGD {
 public:
  SGD(float learning_rate) : learning_rate_(new Variable("sgd")) {
    Tape init_tape;

    std::string initializer = "fill_constant";
    framework::AttributeMap attrs;
    attrs["dtype"] = paddle::framework::proto::VarType::Type::VarType_Type_FP32;
    attrs["shape"] = std::vector<int>{1};
    attrs["value"] = learning_rate;
    init_tape.AddOp(initializer, {}, {{"Out", {learning_rate_}}}, attrs);

    init_tape.Forward();
  }

  void operator()(VariableHandle input) {
    Tape temp_tape;
    temp_tape.AddOp("sgd",
                    {{"Param", {input}},
                     {"LearningRate", {learning_rate_}},
                     {"Grad", {input->Grad()}}},
                    {{"ParamOut", {input}}},
                    {});
    temp_tape.Forward();
    input->ResetGrad();
  }

 private:
  VariableHandle learning_rate_;
};
}
}
