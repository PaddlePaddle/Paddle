// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

namespace paddle {
namespace distributed {
namespace auto_parallel {

#define REGISTER_SPMD_RULE(op_type, rule_class, ...)                        \
  STATIC_ASSERT_GLOBAL_NAMESPACE(                                           \
      __reg_spmd_rule_##op_type,                                            \
      "REGISTER_SPMD_RULE must be called in global namespace");             \
  int __holder_##op_type =                                                  \
      ::paddle::distributed::auto_parallel::SPMDRuleMap::Instance().Insert( \
          #op_type, std::make_unique<rule_class>(__VA_ARGS__))

REGISTER_SPMD_RULE(matmul, paddle::distributed::auto_parallel::MatmulSPMDRule);

}  // namespace auto_parallel
}  // namespace distributed
}  // namespace paddle
