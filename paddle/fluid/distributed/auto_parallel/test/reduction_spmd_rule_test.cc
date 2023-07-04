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
#include "paddle/fluid/distributed/auto_parallel/spmd_rules/reduction_spmd_rule.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/process_mesh.h"

namespace paddle {
namespace distributed {
namespace auto_parallel {

TEST(ReductionSMPDRuleTest, SingleMeshDim) {
  // build input data class
  std::vector<int64_t> x_shape = {64, 36};

  std::vector<int64_t> mesh_shape = {4};
  std::vector<int64_t> process_ids = {0, 1, 2, 3};
  std::vector<std::string> dim_names = {"x"};
  ProcessMesh process_mesh(mesh_shape, process_ids, dim_names);

  TensorDistAttr x_dist_attr = TensorDistAttr();
  x_dist_attr.set_process_mesh(process_mesh);
  x_dist_attr.set_batch_dim(-1);
  x_dist_attr.set_dynamic_dims(std::vector<bool>({false, false}));

  DistTensorSpec x_dist_tensor_spec = DistTensorSpec(x_shape, x_dist_attr);
  paddle::framework::AttributeMap attrs;
  ReductionSPMDRule reduction_rule = ReductionSPMDRule();
  size_t input_size = 1, output_size = 1;

  attrs["linearity"] = false;

  // reduce on dim 0, keep_dim = false
  // [0, -1] --> [-1, -1], [-1]
  attrs["keep_dim"] = false;
  attrs["axis"] = std::vector<int64_t>({0});
  x_dist_tensor_spec.set_dims_mapping({0, -1});
  std::pair<std::vector<TensorDistAttr>, std::vector<TensorDistAttr>>
      infered_dist_attrs =
          reduction_rule.InferForward({x_dist_tensor_spec}, attrs);
  EXPECT_EQ(infered_dist_attrs.first.size(), input_size);
  EXPECT_EQ(infered_dist_attrs.second.size(), output_size);
  EXPECT_EQ(infered_dist_attrs.first[0].dims_mapping(),
            std::vector<int64_t>({-1, -1}));
  EXPECT_EQ(infered_dist_attrs.second[0].dims_mapping(),
            std::vector<int64_t>({-1}));
  VLOG(4) << "SingleMeshDim test1 done." << std::endl << std::endl << std::endl;

  // reduce on dim 0, keep_dim = true
  // [0, -1] --> [-1, -1], [-1, -1]
  attrs["keep_dim"] = true;
  x_dist_tensor_spec.set_dims_mapping({0, -1});
  infered_dist_attrs = reduction_rule.InferForward({x_dist_tensor_spec}, attrs);
  EXPECT_EQ(infered_dist_attrs.first.size(), input_size);
  EXPECT_EQ(infered_dist_attrs.second.size(), output_size);
  EXPECT_EQ(infered_dist_attrs.first[0].dims_mapping(),
            std::vector<int64_t>({-1, -1}));
  EXPECT_EQ(infered_dist_attrs.second[0].dims_mapping(),
            std::vector<int64_t>({-1, -1}));
  VLOG(4) << "SingleMeshDim test2 done." << std::endl << std::endl << std::endl;

  // reduce on dim 1, keep_dim = false
  // [0, -1] --> [0, -1], [0]
  x_dist_tensor_spec.set_dims_mapping({0, -1});
  attrs["keep_dim"] = false;
  attrs["axis"] = std::vector<int64_t>({1});
  infered_dist_attrs = reduction_rule.InferForward({x_dist_tensor_spec}, attrs);
  EXPECT_EQ(infered_dist_attrs.first.size(), input_size);
  EXPECT_EQ(infered_dist_attrs.second.size(), output_size);
  EXPECT_EQ(infered_dist_attrs.first[0].dims_mapping(),
            std::vector<int64_t>({0, -1}));
  EXPECT_EQ(infered_dist_attrs.second[0].dims_mapping(),
            std::vector<int64_t>({0}));
  VLOG(4) << "SingleMeshDim test3 done." << std::endl << std::endl << std::endl;

  // reduce on dim 1, keep_dim = false
  // [0, -1] --> [0, -1], [0]
  x_dist_tensor_spec.set_dims_mapping({0, -1});
  attrs["keep_dim"] = false;
  attrs["axis"] = std::vector<int64_t>({-1});
  infered_dist_attrs = reduction_rule.InferForward({x_dist_tensor_spec}, attrs);
  EXPECT_EQ(infered_dist_attrs.first.size(), input_size);
  EXPECT_EQ(infered_dist_attrs.second.size(), output_size);
  EXPECT_EQ(infered_dist_attrs.first[0].dims_mapping(),
            std::vector<int64_t>({0, -1}));
  EXPECT_EQ(infered_dist_attrs.second[0].dims_mapping(),
            std::vector<int64_t>({0}));
  VLOG(4) << "SingleMeshDim test4 done." << std::endl << std::endl << std::endl;

  // reduce on dim 1, keep_dim = true
  // [0, -1] --> [0, -1], [0, -1]
  x_dist_tensor_spec.set_dims_mapping({0, -1});
  attrs["keep_dim"] = true;
  attrs["axis"] = std::vector<int64_t>({1});
  infered_dist_attrs = reduction_rule.InferForward({x_dist_tensor_spec}, attrs);
  EXPECT_EQ(infered_dist_attrs.first.size(), input_size);
  EXPECT_EQ(infered_dist_attrs.second.size(), output_size);
  EXPECT_EQ(infered_dist_attrs.first[0].dims_mapping(),
            std::vector<int64_t>({0, -1}));
  EXPECT_EQ(infered_dist_attrs.second[0].dims_mapping(),
            std::vector<int64_t>({0, -1}));
  VLOG(4) << "SingleMeshDim test5 done." << std::endl << std::endl << std::endl;

  // reduce on dim 0 and 1, keep_dim = false
  // [0, -1] --> [-1, -1], []
  x_dist_tensor_spec.set_dims_mapping({0, -1});
  attrs["keep_dim"] = false;
  attrs["axis"] = std::vector<int64_t>({0, 1});
  infered_dist_attrs = reduction_rule.InferForward({x_dist_tensor_spec}, attrs);
  input_size = 1;
  output_size = 1;
  EXPECT_EQ(infered_dist_attrs.first.size(), input_size);
  EXPECT_EQ(infered_dist_attrs.second.size(), output_size);
  EXPECT_EQ(infered_dist_attrs.first[0].dims_mapping(),
            std::vector<int64_t>({-1, -1}));
  EXPECT_EQ(infered_dist_attrs.second[0].dims_mapping().size(), (size_t)0);
  VLOG(4) << "SingleMeshDim test6 done." << std::endl << std::endl << std::endl;

  // reduce on dim 0 and 1, keep_dim = true
  // [0, -1] --> [-1, -1], [-1, -1]
  x_dist_tensor_spec.set_dims_mapping({0, -1});
  attrs["keep_dim"] = true;
  attrs["axis"] = std::vector<int64_t>({0, 1});
  infered_dist_attrs = reduction_rule.InferForward({x_dist_tensor_spec}, attrs);
  input_size = 1;
  output_size = 1;
  EXPECT_EQ(infered_dist_attrs.first.size(), input_size);
  EXPECT_EQ(infered_dist_attrs.second.size(), output_size);
  EXPECT_EQ(infered_dist_attrs.first[0].dims_mapping(),
            std::vector<int64_t>({-1, -1}));
  EXPECT_EQ(infered_dist_attrs.second[0].dims_mapping(),
            std::vector<int64_t>({-1, -1}));
  VLOG(4) << "SingleMeshDim test7 done." << std::endl << std::endl << std::endl;

  // test linear ops that support partial state
  attrs["linearity"] = true;

  // reduce on dim 0, keep_dim = false
  // [0, -1] --> [0, -1], [-1], partial_on_dim = [0]
  attrs["keep_dim"] = false;
  attrs["axis"] = std::vector<int64_t>({0});
  x_dist_tensor_spec.set_dims_mapping({0, -1});
  infered_dist_attrs = reduction_rule.InferForward({x_dist_tensor_spec}, attrs);
  EXPECT_EQ(infered_dist_attrs.first.size(), input_size);
  EXPECT_EQ(infered_dist_attrs.second.size(), output_size);
  EXPECT_EQ(infered_dist_attrs.first[0].dims_mapping(),
            std::vector<int64_t>({0, -1}));
  EXPECT_EQ(infered_dist_attrs.second[0].dims_mapping(),
            std::vector<int64_t>({-1}));
  VLOG(4) << "SingleMeshDim test8 done." << std::endl << std::endl << std::endl;

  // reduce on dim 1, keep_dim = false
  // [0, -1] --> [0, -1], [0], partial_on_dim = []
  attrs["axis"] = std::vector<int64_t>({1});
  x_dist_tensor_spec.set_dims_mapping({0, -1});
  infered_dist_attrs = reduction_rule.InferForward({x_dist_tensor_spec}, attrs);
  EXPECT_EQ(infered_dist_attrs.first.size(), input_size);
  EXPECT_EQ(infered_dist_attrs.second.size(), output_size);
  EXPECT_EQ(infered_dist_attrs.first[0].dims_mapping(),
            std::vector<int64_t>({0, -1}));
  EXPECT_EQ(infered_dist_attrs.second[0].dims_mapping(),
            std::vector<int64_t>({0}));
  VLOG(4) << "SingleMeshDim test9 done." << std::endl << std::endl << std::endl;
}

TEST(ReductionSMPDRuleTest, MultiMeshDim) {
  std::vector<int64_t> mesh_shape = {2, 3};
  std::vector<int64_t> process_ids = {0, 1, 2, 3, 4, 5};
  std::vector<std::string> dim_names = {"x", "y"};
  ProcessMesh process_mesh(mesh_shape, process_ids, dim_names);

  TensorDistAttr x_dist_attr = TensorDistAttr();
  x_dist_attr.set_process_mesh(process_mesh);
  x_dist_attr.set_batch_dim(-1);
  x_dist_attr.set_dynamic_dims(std::vector<bool>({false, false}));

  DistTensorSpec x_dist_tensor_spec = DistTensorSpec({96, 24, 48}, x_dist_attr);
  std::vector<int64_t> x_shape = x_dist_tensor_spec.shape();

  paddle::framework::AttributeMap attrs;
  ReductionSPMDRule reduction_rule = ReductionSPMDRule();
  std::pair<std::vector<TensorDistAttr>, std::vector<TensorDistAttr>>
      infered_dist_attrs;
  attrs["linearity"] = false;

  // reduce on dim 1, 2, keep_dim = false
  // [0, -1, -1] --> [0, -1, -1], [0]
  x_dist_tensor_spec.set_dims_mapping({0, -1, -1});
  attrs["keep_dim"] = false;
  attrs["axis"] = std::vector<int64_t>({1, 2});
  infered_dist_attrs = reduction_rule.InferForward({x_dist_tensor_spec}, attrs);
  size_t input_size = 1, output_size = 1;
  EXPECT_EQ(infered_dist_attrs.first.size(), input_size);
  EXPECT_EQ(infered_dist_attrs.second.size(), output_size);
  EXPECT_EQ(infered_dist_attrs.first[0].dims_mapping(),
            std::vector<int64_t>({0, -1, -1}));
  EXPECT_EQ(infered_dist_attrs.second[0].dims_mapping(),
            std::vector<int64_t>({0}));
  VLOG(4) << "MultiMeshDim test1 done." << std::endl << std::endl << std::endl;

  // reduce on dim 1, 2, keep_dim = false
  // [-1, 0, 1] --> [-1, -1, -1], [-1]
  x_dist_tensor_spec.set_dims_mapping({-1, 0, 1});
  attrs["keep_dim"] = false;
  attrs["axis"] = std::vector<int64_t>({1, 2});
  infered_dist_attrs = reduction_rule.InferForward({x_dist_tensor_spec}, attrs);
  EXPECT_EQ(infered_dist_attrs.first.size(), input_size);
  EXPECT_EQ(infered_dist_attrs.second.size(), output_size);
  EXPECT_EQ(infered_dist_attrs.first[0].dims_mapping(),
            std::vector<int64_t>({-1, -1, -1}));
  EXPECT_EQ(infered_dist_attrs.second[0].dims_mapping(),
            std::vector<int64_t>({-1}));
  VLOG(4) << "MultiMeshDim test2 done." << std::endl << std::endl << std::endl;

  // reduction on dim 1, 2, keep_dim = false
  // [1, -1, -1] --> [1, -1, -1], [1]
  x_dist_tensor_spec.set_dims_mapping({1, -1, -1});
  attrs["keep_dim"] = false;
  attrs["axis"] = std::vector<int64_t>({1, 2});
  infered_dist_attrs = reduction_rule.InferForward({x_dist_tensor_spec}, attrs);
  EXPECT_EQ(infered_dist_attrs.first.size(), input_size);
  EXPECT_EQ(infered_dist_attrs.second.size(), output_size);
  EXPECT_EQ(infered_dist_attrs.first[0].dims_mapping(),
            std::vector<int64_t>({1, -1, -1}));
  EXPECT_EQ(infered_dist_attrs.second[0].dims_mapping(),
            std::vector<int64_t>({1}));
  VLOG(4) << "MultiMeshDim test3 done." << std::endl << std::endl << std::endl;

  // reduction on dim 1, 2, keep_dim = false
  // [0, 1, -1] --> [0, -1, -1], [0]
  x_dist_tensor_spec.set_dims_mapping({0, 1, -1});
  attrs["keep_dim"] = false;
  attrs["axis"] = std::vector<int64_t>({1, 2});
  infered_dist_attrs = reduction_rule.InferForward({x_dist_tensor_spec}, attrs);
  EXPECT_EQ(infered_dist_attrs.first.size(), input_size);
  EXPECT_EQ(infered_dist_attrs.second.size(), output_size);
  EXPECT_EQ(infered_dist_attrs.first[0].dims_mapping(),
            std::vector<int64_t>({0, -1, -1}));
  EXPECT_EQ(infered_dist_attrs.second[0].dims_mapping(),
            std::vector<int64_t>({0}));
  VLOG(4) << "MultiMeshDim test4 done." << std::endl << std::endl << std::endl;

  // reduction on dim 1, 2, keep_dim = false
  // [0, 1, -1] --> [0, -1, -1], [0]
  x_dist_tensor_spec.set_dims_mapping({0, 1, -1});
  attrs["keep_dim"] = false;
  attrs["axis"] = std::vector<int64_t>({-2, -1});
  infered_dist_attrs = reduction_rule.InferForward({x_dist_tensor_spec}, attrs);
  EXPECT_EQ(infered_dist_attrs.first.size(), input_size);
  EXPECT_EQ(infered_dist_attrs.second.size(), output_size);
  EXPECT_EQ(infered_dist_attrs.first[0].dims_mapping(),
            std::vector<int64_t>({0, -1, -1}));
  EXPECT_EQ(infered_dist_attrs.second[0].dims_mapping(),
            std::vector<int64_t>({0}));
  VLOG(4) << "MultiMeshDim test5 done." << std::endl << std::endl << std::endl;

  // reduction on dim 1, 2, keep_dim = true
  // [0, 1, -1] --> [0, -1, -1], [0, -1, -1]
  x_dist_tensor_spec.set_dims_mapping({0, 1, -1});
  attrs["keep_dim"] = true;
  attrs["axis"] = std::vector<int64_t>({-2, -1});
  infered_dist_attrs = reduction_rule.InferForward({x_dist_tensor_spec}, attrs);
  EXPECT_EQ(infered_dist_attrs.first.size(), input_size);
  EXPECT_EQ(infered_dist_attrs.second.size(), output_size);
  EXPECT_EQ(infered_dist_attrs.first[0].dims_mapping(),
            std::vector<int64_t>({0, -1, -1}));
  EXPECT_EQ(infered_dist_attrs.second[0].dims_mapping(),
            std::vector<int64_t>({0, -1, -1}));
  VLOG(4) << "MultiMeshDim test6 done." << std::endl << std::endl << std::endl;

  // test partial states
  attrs["linearity"] = true;

  // reduce on dim 1, 2, keep_dim = false
  // [0, -1, -1] --> [0, -1, -1], [0], parital_on_dim = []
  x_dist_tensor_spec.set_dims_mapping({0, -1, -1});
  attrs["keep_dim"] = false;
  attrs["axis"] = std::vector<int64_t>({1, 2});
  infered_dist_attrs = reduction_rule.InferForward({x_dist_tensor_spec}, attrs);
  EXPECT_EQ(infered_dist_attrs.first.size(), input_size);
  EXPECT_EQ(infered_dist_attrs.second.size(), output_size);
  EXPECT_EQ(infered_dist_attrs.first[0].dims_mapping(),
            std::vector<int64_t>({0, -1, -1}));
  EXPECT_EQ(infered_dist_attrs.second[0].dims_mapping(),
            std::vector<int64_t>({0}));
  VLOG(4) << "MultiMeshDim test7 done." << std::endl << std::endl << std::endl;

  // reduce on dim 1, 2, keep_dim = false
  // [-1, 0, 1] --> [-1, 0, 1], [-1], parital_on_dim = [0, 1]
  x_dist_tensor_spec.set_dims_mapping({-1, 0, 1});
  attrs["keep_dim"] = false;
  attrs["axis"] = std::vector<int64_t>({1, 2});
  infered_dist_attrs = reduction_rule.InferForward({x_dist_tensor_spec}, attrs);
  EXPECT_EQ(infered_dist_attrs.first.size(), input_size);
  EXPECT_EQ(infered_dist_attrs.second.size(), output_size);
  EXPECT_EQ(infered_dist_attrs.first[0].dims_mapping(),
            std::vector<int64_t>({-1, 0, 1}));
  EXPECT_EQ(infered_dist_attrs.second[0].dims_mapping(),
            std::vector<int64_t>({-1}));
  VLOG(4) << "MultiMeshDim test8 done." << std::endl << std::endl << std::endl;
}

}  // namespace auto_parallel
}  // namespace distributed
}  // namespace paddle
