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

namespace cinn {
namespace hlir {
namespace op {

ExternalApiInfo& ExternalApiRegistry::Register(const std::string& op_name,
                                               const common::Target& target) {
  return __REGISTER__(GenKey(op_name, target));
}

std::string ExternalApiRegistry::GetExternalApi(const framework::Node* op_node,
                                                const common::Target& target) {
  CHECK(op_node->attrs.attr_store.count("original_op"))
      << "a custom_call op must store its original op name";
  std::string op_name =
      absl::get<std::string>(op_node->attrs.attr_store.at("original_op"));
  const ExternalApiInfo* external_api_info = Find(GenKey(op_name, target));
  CHECK(external_api_info) << "Op:" << op_name
                           << " doesn't register external_api on " << target;
  std::string external_api = external_api_info->api_name;
  if (external_api.empty()) {  // if api_name not set directly, call trans_func
                               // to acquire
    auto&& trans_func = external_api_info->trans_func;
    CHECK(trans_func) << "Op:" << op_name
                      << " register invalid ExternalApiInfo on " << target;
    external_api = trans_func(op_node);
  }
  return external_api;
}

std::string ExternalApiRegistry::GenKey(const std::string& op_name,
                                        const common::Target& target) {
  std::ostringstream oss;
  oss << target;
  return op_name + "_" + oss.str();
}

}  // namespace op
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(op_external_api) {
  const auto& default_nvgpu = ::cinn::common::DefaultNVGPUTarget();
  const auto& default_host = ::cinn::common::DefaultHostTarget();

  CINN_OP_REGISTER_EXTERNAL_API(matmul, default_nvgpu)
      .set_api_name("cinn_call_cublas");
  CINN_OP_REGISTER_EXTERNAL_API(mul, default_nvgpu)
      .set_api_name("cinn_call_cublas");
  CINN_OP_REGISTER_EXTERNAL_API(cublas_gemm, default_nvgpu)
      .set_api_name("cinn_call_cublas");
  CINN_OP_REGISTER_EXTERNAL_API(cublas_matmul, default_nvgpu)
      .set_api_name("cinn_call_cublas");
  CINN_OP_REGISTER_EXTERNAL_API(gaussian_random, default_nvgpu)
      .set_api_name("cinn_call_gaussian_random");
  CINN_OP_REGISTER_EXTERNAL_API(uniform_random, default_nvgpu)
      .set_api_name("cinn_call_uniform_random");
  CINN_OP_REGISTER_EXTERNAL_API(randint, default_nvgpu)
      .set_api_name("cinn_call_randint");
  CINN_OP_REGISTER_EXTERNAL_API(cholesky, default_nvgpu)
      .set_api_name("cinn_call_cholesky_nvgpu");
  CINN_OP_REGISTER_EXTERNAL_API(cholesky, default_host)
      .set_api_name("cinn_call_cholesky_host");
  CINN_OP_REGISTER_EXTERNAL_API(triangular_solve, default_nvgpu)
      .set_api_name("cinn_call_triangular_solve_nvgpu");
  CINN_OP_REGISTER_EXTERNAL_API(assert_true, default_nvgpu)
      .set_api_name("cinn_assert_true_nvgpu");
  CINN_OP_REGISTER_EXTERNAL_API(assert_true, default_host)
      .set_api_name("cinn_assert_true_host");
#ifdef CINN_WITH_CUDNN
  CINN_OP_REGISTER_EXTERNAL_API(conv2d, default_nvgpu)
      .set_trans_func([](const ::cinn::hlir::framework::Node* node) {
        CHECK(node->attrs.attr_store.count("conv_type"));
        std::string conv_type =
            absl::get<std::string>(node->attrs.attr_store.at("conv_type"));
        CHECK(conv_type == "forward" || conv_type == "backward_data" ||
              conv_type == "backward_filter")
            << "unknown conv_type=" << conv_type;
        return "cinn_call_cudnn_conv2d_" + conv_type;
      });
  CINN_OP_REGISTER_EXTERNAL_API(depthwise_conv2d, default_nvgpu)
      .set_trans_func([](const ::cinn::hlir::framework::Node* node) {
        std::string conv_type =
            node->attrs.attr_store.count("conv_type")
                ? absl::get<std::string>(node->attrs.attr_store.at("conv_type"))
                : "forward";
        CHECK(conv_type == "forward" || conv_type == "backward_data" ||
              conv_type == "backward_filter")
            << "unknown conv_type=" << conv_type;
        return "cinn_call_cudnn_conv2d_" + conv_type;
      });
  CINN_OP_REGISTER_EXTERNAL_API(pool2d, default_nvgpu)
      .set_api_name("cinn_call_cudnn_pool2d_forward");
  CINN_OP_REGISTER_EXTERNAL_API(pool2d_grad, default_nvgpu)
      .set_api_name("cinn_call_cudnn_pool2d_backward");
#endif
  return true;
}
