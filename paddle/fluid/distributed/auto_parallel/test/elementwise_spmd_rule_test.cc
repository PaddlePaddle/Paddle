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

#include <iostream>
#include "gtest/gtest.h"

#include "paddle/fluid/distributed/auto_parallel/spmd_rules/dist_tensor_spec.h"
#include "paddle/fluid/distributed/auto_parallel/spmd_rules/elementwise_spmd_rule.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/process_mesh.h"

namespace paddle {
namespace distributed {
namespace auto_parallel {

TEST(ElementwiseSMPDRuleTest, SingleMeshDim) {
  // build input data class
  std::vector<int64_t> x_shape = {64, 36};
  std::vector<int64_t> y_shape = {64, 36};

  std::vector<int64_t> mesh_shape = {4};
  std::vector<int64_t> process_ids = {0, 1, 2, 3};
  std::vector<std::string> dim_names = {"x"};
  ProcessMesh process_mesh(mesh_shape, process_ids, dim_names);

  TensorDistAttr x_dist_attr = TensorDistAttr();
  x_dist_attr.set_process_mesh(process_mesh);
  x_dist_attr.set_dims_mapping(std::vector<int64_t>({0, -1}));
  x_dist_attr.set_batch_dim(-1);
  x_dist_attr.set_dynamic_dims(std::vector<bool>({false, false}));

  TensorDistAttr y_dist_attr = TensorDistAttr();
  y_dist_attr.set_process_mesh(process_mesh);
  y_dist_attr.set_dims_mapping(std::vector<int64_t>({-1, -1}));
  y_dist_attr.set_batch_dim(-1);
  y_dist_attr.set_dynamic_dims(std::vector<bool>({false, false}));

  DistTensorSpec x_dist_tensor_spec = DistTensorSpec(x_shape, x_dist_attr);
  DistTensorSpec y_dist_tensor_spec = DistTensorSpec(y_shape, y_dist_attr);

  paddle::framework::AttributeMap attrs;

  ElementwiseSPMDRule element_rule = ElementwiseSPMDRule();

  // [0, -1], [-1, -1] --> [0, -1], [0, -1], [0, -1]
  std::pair<std::vector<TensorDistAttr>, std::vector<TensorDistAttr>>
      infered_dist_attrs = element_rule.InferForward(
          {x_dist_tensor_spec, y_dist_tensor_spec}, attrs);
  size_t input_size = 2;
  size_t output_size = 1;
  EXPECT_EQ(infered_dist_attrs.first.size(), input_size);
  EXPECT_EQ(infered_dist_attrs.second.size(), output_size);
  EXPECT_EQ(infered_dist_attrs.first[0].dims_mapping(),
            std::vector<int64_t>({0, -1}));
  EXPECT_EQ(infered_dist_attrs.first[1].dims_mapping(),
            std::vector<int64_t>({0, -1}));
  EXPECT_EQ(infered_dist_attrs.second[0].dims_mapping(),
            std::vector<int64_t>({0, -1}));
  VLOG(4) << "SingleMeshDim test1 done." << std::endl << std::endl << std::endl;

  // [0, -1], [-1, 0] --> [0, -1], [0, -1], [0, -1]
  x_dist_tensor_spec.set_dims_mapping({0, -1});
  y_dist_tensor_spec.set_dims_mapping({-1, 0});
  infered_dist_attrs = element_rule.InferForward(
      {x_dist_tensor_spec, y_dist_tensor_spec}, attrs);
  EXPECT_EQ(infered_dist_attrs.first[0].dims_mapping(),
            std::vector<int64_t>({0, -1}));
  EXPECT_EQ(infered_dist_attrs.first[1].dims_mapping(),
            std::vector<int64_t>({0, -1}));
  EXPECT_EQ(infered_dist_attrs.second[0].dims_mapping(),
            std::vector<int64_t>({0, -1}));
  VLOG(4) << "SingleMeshDim test2 done." << std::endl << std::endl << std::endl;

  // [-1, 0]--> [-1, 0], [-1, 0]
  x_dist_tensor_spec.set_dims_mapping({-1, 0});
  infered_dist_attrs = element_rule.InferForward({x_dist_tensor_spec}, attrs);
  EXPECT_EQ(infered_dist_attrs.first[0].dims_mapping(),
            std::vector<int64_t>({-1, 0}));
  EXPECT_EQ(infered_dist_attrs.second[0].dims_mapping(),
            std::vector<int64_t>({-1, 0}));
  VLOG(4) << "SingleMeshDim test3 done." << std::endl << std::endl << std::endl;
}

TEST(ElementwiseSMPDRuleTest, SingleMeshDimBroadcast) {
  // build input data class
  std::vector<int64_t> x_shape = {64, 36, 12};
  std::vector<int64_t> y_shape = {12};

  std::vector<int64_t> mesh_shape = {4};
  std::vector<int64_t> process_ids = {0, 1, 2, 3};
  std::vector<std::string> dim_names = {"x"};
  ProcessMesh process_mesh(mesh_shape, process_ids, dim_names);

  TensorDistAttr x_dist_attr = TensorDistAttr();
  x_dist_attr.set_process_mesh(process_mesh);
  x_dist_attr.set_batch_dim(-1);
  x_dist_attr.set_dynamic_dims(std::vector<bool>({false, false}));

  TensorDistAttr y_dist_attr = TensorDistAttr();
  y_dist_attr.set_process_mesh(process_mesh);
  y_dist_attr.set_batch_dim(-1);
  y_dist_attr.set_dynamic_dims(std::vector<bool>({false, false}));

  DistTensorSpec x_dist_tensor_spec = DistTensorSpec(x_shape, x_dist_attr);
  DistTensorSpec y_dist_tensor_spec = DistTensorSpec(y_shape, y_dist_attr);

  paddle::framework::AttributeMap attrs;

  ElementwiseSPMDRule element_rule = ElementwiseSPMDRule();

  size_t input_size = 2;
  size_t output_size = 1;

  // [0, -1, -1], [-1] --> [0, -1, -1], [-1], [0, -1, -1]
  x_dist_tensor_spec.set_dims_mapping({0, -1, -1});
  y_dist_tensor_spec.set_dims_mapping({-1});
  std::pair<std::vector<TensorDistAttr>, std::vector<TensorDistAttr>>
      infered_dist_attrs = element_rule.InferForward(
          {x_dist_tensor_spec, y_dist_tensor_spec}, attrs);
  EXPECT_EQ(infered_dist_attrs.first.size(), input_size);
  EXPECT_EQ(infered_dist_attrs.second.size(), output_size);
  EXPECT_EQ(infered_dist_attrs.first[0].dims_mapping(),
            std::vector<int64_t>({0, -1, -1}));
  EXPECT_EQ(infered_dist_attrs.first[1].dims_mapping(),
            std::vector<int64_t>({-1}));
  EXPECT_EQ(infered_dist_attrs.second[0].dims_mapping(),
            std::vector<int64_t>({0, -1, -1}));
  VLOG(4) << "SingleMeshDimBroadcast test1 done." << std::endl
          << std::endl
          << std::endl;

  // [-1, -1, -1], [0] --> [-1, -1, 0], [0], [-1, -1, 0]
  x_dist_tensor_spec.set_dims_mapping({-1, -1, -1});
  y_dist_tensor_spec.set_dims_mapping({0});
  infered_dist_attrs = element_rule.InferForward(
      {x_dist_tensor_spec, y_dist_tensor_spec}, attrs);
  EXPECT_EQ(infered_dist_attrs.first.size(), input_size);
  EXPECT_EQ(infered_dist_attrs.second.size(), output_size);
  EXPECT_EQ(infered_dist_attrs.first[0].dims_mapping(),
            std::vector<int64_t>({-1, -1, 0}));
  EXPECT_EQ(infered_dist_attrs.first[1].dims_mapping(),
            std::vector<int64_t>({0}));
  EXPECT_EQ(infered_dist_attrs.second[0].dims_mapping(),
            std::vector<int64_t>({-1, -1, 0}));
  VLOG(4) << "SingleMeshDimBroadcast test2 done." << std::endl
          << std::endl
          << std::endl;

  x_dist_tensor_spec.set_shape({64, 1, 1, 12});
  y_dist_tensor_spec.set_shape({64, 32, 12});
  // [0, -1, -1, -1], [-1, -1, -1] --> [0, -1, -1, -1], [-1, -1, -1], [0, -1,
  // -1, -1]
  x_dist_tensor_spec.set_dims_mapping({0, -1, -1, -1});
  y_dist_tensor_spec.set_dims_mapping({-1, -1, -1});
  infered_dist_attrs = element_rule.InferForward(
      {x_dist_tensor_spec, y_dist_tensor_spec}, attrs);
  EXPECT_EQ(infered_dist_attrs.first.size(), input_size);
  EXPECT_EQ(infered_dist_attrs.second.size(), output_size);
  EXPECT_EQ(infered_dist_attrs.first[0].dims_mapping(),
            std::vector<int64_t>({0, -1, -1, -1}));
  EXPECT_EQ(infered_dist_attrs.first[1].dims_mapping(),
            std::vector<int64_t>({-1, -1, -1}));
  EXPECT_EQ(infered_dist_attrs.second[0].dims_mapping(),
            std::vector<int64_t>({0, -1, -1, -1}));
  VLOG(4) << "SingleMeshDimBroadcast test3 done." << std::endl
          << std::endl
          << std::endl;
}

TEST(ElementwiseSMPDRuleTest, MultiMeshDim) {
  std::vector<int64_t> mesh_shape = {2, 3};
  std::vector<int64_t> process_ids = {0, 1, 2, 3, 4, 5};
  std::vector<std::string> dim_names = {"x", "y"};
  ProcessMesh process_mesh(mesh_shape, process_ids, dim_names);

  TensorDistAttr x_dist_attr = TensorDistAttr();
  x_dist_attr.set_process_mesh(process_mesh);
  x_dist_attr.set_batch_dim(-1);
  x_dist_attr.set_dynamic_dims(std::vector<bool>({false, false}));

  TensorDistAttr y_dist_attr = TensorDistAttr();
  y_dist_attr.set_process_mesh(process_mesh);
  y_dist_attr.set_batch_dim(-1);
  y_dist_attr.set_dynamic_dims(std::vector<bool>({false, false}));

  DistTensorSpec x_dist_tensor_spec = DistTensorSpec({96, 24, 48}, x_dist_attr);
  DistTensorSpec y_dist_tensor_spec = DistTensorSpec({96, 24, 48}, y_dist_attr);

  paddle::framework::AttributeMap attrs;
  ElementwiseSPMDRule element_rule = ElementwiseSPMDRule();
  std::pair<std::vector<TensorDistAttr>, std::vector<TensorDistAttr>>
      infered_dist_attrs;

  size_t input_size = 2;
  size_t output_size = 1;

  // [0, 1, -1], [-1, -1, -1] --> [0, 1, -1], [0, 1, -1], [0, 1, -1]
  x_dist_tensor_spec.set_dims_mapping({0, 1, -1});
  y_dist_tensor_spec.set_dims_mapping({-1, -1, -1});
  infered_dist_attrs = element_rule.InferForward(
      {x_dist_tensor_spec, y_dist_tensor_spec}, attrs);
  EXPECT_EQ(infered_dist_attrs.first.size(), input_size);
  EXPECT_EQ(infered_dist_attrs.second.size(), output_size);
  EXPECT_EQ(infered_dist_attrs.first[0].dims_mapping(),
            std::vector<int64_t>({0, 1, -1}));
  EXPECT_EQ(infered_dist_attrs.first[1].dims_mapping(),
            std::vector<int64_t>({0, 1, -1}));
  EXPECT_EQ(infered_dist_attrs.second[0].dims_mapping(),
            std::vector<int64_t>({0, 1, -1}));
  VLOG(4) << "MultiMeshDim test1 done." << std::endl << std::endl << std::endl;

  // [0, -1, -1], [-1, 1, 0] --> [0, 1, -1], [0, 1, -1], [0, 1, -1]
  x_dist_tensor_spec.set_dims_mapping({0, -1, -1});
  y_dist_tensor_spec.set_dims_mapping({-1, 1, 0});
  infered_dist_attrs = element_rule.InferForward(
      {x_dist_tensor_spec, y_dist_tensor_spec}, attrs);
  EXPECT_EQ(infered_dist_attrs.first.size(), input_size);
  EXPECT_EQ(infered_dist_attrs.second.size(), output_size);
  EXPECT_EQ(infered_dist_attrs.first[0].dims_mapping(),
            std::vector<int64_t>({0, 1, -1}));
  EXPECT_EQ(infered_dist_attrs.first[1].dims_mapping(),
            std::vector<int64_t>({0, 1, -1}));
  EXPECT_EQ(infered_dist_attrs.second[0].dims_mapping(),
            std::vector<int64_t>({0, 1, -1}));
  VLOG(4) << "MultiMeshDim test2 done." << std::endl << std::endl << std::endl;
}

TEST(ElementwiseSMPDRuleTest, MultiMeshDimBroadcast) {
  std::vector<int64_t> mesh_shape = {2, 3};
  std::vector<int64_t> process_ids = {0, 1, 2, 3, 4, 5};
  std::vector<std::string> dim_names = {"x", "y"};
  ProcessMesh process_mesh(mesh_shape, process_ids, dim_names);

  TensorDistAttr x_dist_attr = TensorDistAttr();
  x_dist_attr.set_process_mesh(process_mesh);
  x_dist_attr.set_batch_dim(-1);
  x_dist_attr.set_dynamic_dims(std::vector<bool>({false, false}));

  TensorDistAttr y_dist_attr = TensorDistAttr();
  y_dist_attr.set_process_mesh(process_mesh);
  y_dist_attr.set_batch_dim(-1);
  y_dist_attr.set_dynamic_dims(std::vector<bool>({false, false}));

  DistTensorSpec x_dist_tensor_spec = DistTensorSpec({96, 24, 48}, x_dist_attr);
  DistTensorSpec y_dist_tensor_spec = DistTensorSpec({48}, y_dist_attr);

  paddle::framework::AttributeMap attrs;
  ElementwiseSPMDRule element_rule = ElementwiseSPMDRule();
  std::pair<std::vector<TensorDistAttr>, std::vector<TensorDistAttr>>
      infered_dist_attrs;

  size_t input_size = 2;
  size_t output_size = 1;

  // [0, -1, -1], [1] --> [0, -1, 1], [1], [0, -1, 1]
  x_dist_tensor_spec.set_dims_mapping({0, -1, -1});
  y_dist_tensor_spec.set_dims_mapping({1});
  infered_dist_attrs = element_rule.InferForward(
      {x_dist_tensor_spec, y_dist_tensor_spec}, attrs);
  EXPECT_EQ(infered_dist_attrs.first.size(), input_size);
  EXPECT_EQ(infered_dist_attrs.second.size(), output_size);
  EXPECT_EQ(infered_dist_attrs.first[0].dims_mapping(),
            std::vector<int64_t>({0, -1, 1}));
  EXPECT_EQ(infered_dist_attrs.first[1].dims_mapping(),
            std::vector<int64_t>({1}));
  EXPECT_EQ(infered_dist_attrs.second[0].dims_mapping(),
            std::vector<int64_t>({0, -1, 1}));
  VLOG(4) << "MultiMeshDimBroadcast test1 done." << std::endl
          << std::endl
          << std::endl;

  x_dist_tensor_spec.set_shape({96, 1, 1, 48});
  y_dist_tensor_spec.set_shape({96, 24, 48});
  // [-1, -1, -1, 1], [0, -1, 1] --> [-1, -1, -1, 1], [0, -1, 1], [-1, 0, -1, 1]
  x_dist_tensor_spec.set_dims_mapping({-1, -1, -1, 1});
  y_dist_tensor_spec.set_dims_mapping({0, -1, 1});
  infered_dist_attrs = element_rule.InferForward(
      {x_dist_tensor_spec, y_dist_tensor_spec}, attrs);
  EXPECT_EQ(infered_dist_attrs.first.size(), input_size);
  EXPECT_EQ(infered_dist_attrs.second.size(), output_size);
  EXPECT_EQ(infered_dist_attrs.first[0].dims_mapping(),
            std::vector<int64_t>({-1, -1, -1, 1}));
  EXPECT_EQ(infered_dist_attrs.first[1].dims_mapping(),
            std::vector<int64_t>({0, -1, 1}));
  EXPECT_EQ(infered_dist_attrs.second[0].dims_mapping(),
            std::vector<int64_t>({-1, 0, -1, 1}));
  VLOG(4) << "MultiMeshDimBroadcast test4 done." << std::endl
          << std::endl
          << std::endl;
}

}  // namespace auto_parallel
}  // namespace distributed
}  // namespace paddle
