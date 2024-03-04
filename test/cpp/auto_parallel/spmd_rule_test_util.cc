/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "test/cpp/auto_parallel/spmd_rule_test_util.h"

namespace paddle {
namespace distributed {
namespace auto_parallel {

const std::vector<int64_t>& get_dims_mapping(
    const phi::distributed::ArgDistAttr& dist_attr) {
  EXPECT_TRUE(
      paddle::holds_alternative<phi::distributed::TensorDistAttr>(dist_attr));
  const auto& tensor_attr = paddle::get<0>(dist_attr);
  return tensor_attr.dims_mapping();
}

bool is_partial(const phi::distributed::ArgDistAttr& dist_attr) {
  EXPECT_TRUE(
      paddle::holds_alternative<phi::distributed::TensorDistAttr>(dist_attr));
  const auto& tensor_attr = paddle::get<0>(dist_attr);
  return tensor_attr.is_partial();
}

const std::set<int64_t> get_partial_dims(
    const phi::distributed::ArgDistAttr& dist_attr) {
  EXPECT_TRUE(
      paddle::holds_alternative<phi::distributed::TensorDistAttr>(dist_attr));
  const auto& tensor_attr = paddle::get<0>(dist_attr);
  return tensor_attr.partial_dims();
}

void check_dim_mapping(const phi::distributed::ArgDistAttr& dist_attr,
                       const std::vector<int64_t>& dim_mapping,
                       const std::string& line) {
  EXPECT_TRUE(
      paddle::holds_alternative<phi::distributed::TensorDistAttr>(dist_attr))
      << line;
  EXPECT_EQ(get_dims_mapping(dist_attr), dim_mapping) << line;
}

void check_partial_dims(const phi::distributed::ArgDistAttr& dist_attr,
                        const std::set<int64_t>& dims,
                        const std::string& line) {
  EXPECT_TRUE(
      paddle::holds_alternative<phi::distributed::TensorDistAttr>(dist_attr))
      << line;
  EXPECT_EQ(get_partial_dims(dist_attr), dims) << line;
}

void clean_partial_status(phi::distributed::ArgDistAttr* dist_attr) {
  EXPECT_TRUE(
      paddle::holds_alternative<phi::distributed::TensorDistAttr>(*dist_attr));
  auto& tensor_attr = paddle::get<0>(*dist_attr);
  tensor_attr.clean_partial_status();
}

void clean_partial_dims(phi::distributed::ArgDistAttr* dist_attr,
                        std::vector<int64_t> dims) {
  EXPECT_TRUE(
      paddle::holds_alternative<phi::distributed::TensorDistAttr>(*dist_attr));
  auto& tensor_attr = paddle::get<0>(*dist_attr);
  tensor_attr.clean_partial_dims(dims);
}

void set_partial_status(phi::distributed::ArgDistAttr* dist_attr,
                        std::vector<int64_t> dims) {
  EXPECT_TRUE(
      paddle::holds_alternative<phi::distributed::TensorDistAttr>(*dist_attr));
  auto& tensor_attr = paddle::get<0>(*dist_attr);
  tensor_attr.set_partial_status(dims);
}

}  // namespace auto_parallel
}  // namespace distributed
}  // namespace paddle
