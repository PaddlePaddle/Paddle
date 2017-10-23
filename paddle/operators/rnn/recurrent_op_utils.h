/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#pragma once

#include <string>

#include "paddle/framework/operator.h"

namespace paddle {
namespace operators {
namespace rnn {

using Scope = framework::Scope;

/**
 * Memory of a RNN (same as the role of `Momory` in PaddlePaddle).
 *
 * Memory attributes cached by this op, dims will be infered from
 * boot memories in father scope. Other attributes are copied from Op's proto
 * attributes.
 */
struct StateAttr {
  // name of current state variable
  std::string var;
  // name of previous step's state variable
  std::string pre_var;
  // name of the variables to init this memory (same role of `boot_layer` in
  // PaddlePaddle), which is store in father's scope.
  std::string boot_var;
};

struct Argument {
  std::string step_net;
  std::string step_scopes;
  std::vector<std::string> inlinks;
  std::vector<std::string> outlinks;
  std::vector<rnn::StateAttr> states;
};

struct ArgumentName {
  std::string step_net;
  std::string step_scopes;
  std::string inlinks;
  std::string outlinks;
  std::string states;          // the memory name
  std::string ex_states;       // the previous memory name
  std::string initial_states;  // the boot memory name
};

/**
 * Prepare inputs for each step net.
 */
void SegmentInputs(const std::vector<Scope*>& step_scopes,
                   const std::vector<std::string>& inlinks,
                   const size_t seq_len);

/**
 * Process outputs of step nets and merge to variables.
 */
void ConcatOutputs(const std::vector<Scope*>& step_scopes,
                   const std::vector<std::string>& outlinks,
                   const size_t seq_len, const platform::DeviceContext& ctx);

void LinkMemories(const std::vector<Scope*>& step_scopes,
                  const std::vector<StateAttr>& memories, const size_t step_id,
                  const int offset);

void InitArgument(const ArgumentName& name, Argument* arg,
                  const framework::OperatorBase& op, bool is_grad = false);

}  // namespace rnn
}  // namespace operators
}  // namespace paddle
