// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/hlir/op/external_api_registry.h"

#include <gtest/gtest.h>

#include "paddle/cinn/hlir/framework/node.h"
#include "paddle/cinn/hlir/framework/op.h"

namespace cinn {
namespace hlir {
namespace framework {

using cinn::hlir::framework::Node;
using cinn::hlir::op::ExternalApiRegistry;

TEST(ExternalApiRegistry, Has) {
  ASSERT_TRUE(ExternalApiRegistry::Global()->Has("matmul",
                                                 common::DefaultNVGPUTarget()));
  ASSERT_TRUE(ExternalApiRegistry::Global()->Has("cholesky",
                                                 common::DefaultHostTarget()));
  ASSERT_FALSE(ExternalApiRegistry::Global()->Has(
      "op_doesn't_exist", common::DefaultNVGPUTarget()));
}

TEST(ExternalApiRegistry, GetExternalApi) {
  auto node =
      std::make_unique<Node>(Operator::Get("custom_call"), "custom_call");
  node->attrs.attr_store["original_op"] = std::string("matmul");
  ASSERT_EQ("cinn_call_cublas",
            ExternalApiRegistry::Global()->GetExternalApi(
                node.get(), common::DefaultNVGPUTarget()));
#ifdef CINN_WITH_CUDNN
  node->attrs.attr_store["conv_type"] = std::string("backward_data");
  node->attrs.attr_store["original_op"] = std::string("conv2d");
  ASSERT_EQ("cinn_call_cudnn_conv2d_backward_data",
            ExternalApiRegistry::Global()->GetExternalApi(
                node.get(), common::DefaultNVGPUTarget()));
#endif
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
