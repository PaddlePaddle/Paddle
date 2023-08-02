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
#include <sstream>
#include "gtest/gtest.h"

#include "paddle/fluid/distributed/auto_parallel/spmd_rules/common.h"
#include "paddle/fluid/distributed/auto_parallel/spmd_rules/dist_tensor_spec.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/process_mesh.h"

namespace paddle {
namespace distributed {
namespace auto_parallel {

TEST(MatmulSPMDRule, Ctor) {
  // build input data class
  std::vector<int64_t> x_shape = {64, 32};
  std::vector<int64_t> y_shape = {32, 48};

  std::vector<int64_t> mesh_shape = {2, 3};
  std::vector<int64_t> process_ids = {0, 1, 2, 3, 4, 5};
  std::vector<std::string> dim_names = {"x", "y"};
  ProcessMesh process_mesh(mesh_shape, process_ids, dim_names);

  TensorDistAttr x_dist_attr = TensorDistAttr();
  x_dist_attr.set_process_mesh(process_mesh);
  x_dist_attr.set_dims_mapping(std::vector<int64_t>({1, -1}));
  x_dist_attr.set_dynamic_dims(std::vector<bool>({false, false}));

  TensorDistAttr y_dist_attr = TensorDistAttr();
  y_dist_attr.set_process_mesh(process_mesh);
  y_dist_attr.set_dims_mapping(std::vector<int64_t>({-1, -1}));
  y_dist_attr.set_dynamic_dims(std::vector<bool>({false, false}));

  DistTensorSpec x_dist_tensor_spec = DistTensorSpec(x_shape, x_dist_attr);
  DistTensorSpec y_dist_tensor_spec = DistTensorSpec(y_shape, y_dist_attr);

  paddle::framework::AttributeMap attrs;
  attrs["trans_x"] = false;
  attrs["trans_y"] = false;

  SPMDRuleBase* matmul_rule = SPMDRuleMap::Instance().Get("matmul");

  // mk[1, -1],kn[-1, -1] --> mk[1, -1],kn[-1, -1] = nm[1, -1] partial[]
  std::pair<std::vector<TensorDistAttr>, std::vector<TensorDistAttr>>
      infered_dist_attrs = matmul_rule->InferForward(
          {x_dist_tensor_spec, y_dist_tensor_spec}, attrs);

  size_t input_size = 2;
  size_t output_size = 1;
  EXPECT_EQ(infered_dist_attrs.first.size(), input_size);
  EXPECT_EQ(infered_dist_attrs.second.size(), output_size);

  EXPECT_EQ(infered_dist_attrs.first[0].dims_mapping(),
            std::vector<int64_t>({1, -1}));
  EXPECT_EQ(infered_dist_attrs.first[1].dims_mapping(),
            std::vector<int64_t>({-1, -1}));
  EXPECT_EQ(infered_dist_attrs.second[0].dims_mapping(),
            std::vector<int64_t>({1, -1}));
  EXPECT_EQ(infered_dist_attrs.second[0].is_partial(), false);
  VLOG(4) << "test1 done." << std::endl << std::endl << std::endl;

  // mk[-1,-1],kn[-1,0] --> mk[-1,-1],kn[-1,0] = nm[-1,0] partial[]
  x_dist_tensor_spec.set_dims_mapping({-1, -1});
  y_dist_tensor_spec.set_dims_mapping({-1, 0});
  infered_dist_attrs = matmul_rule->InferForward(
      {x_dist_tensor_spec, y_dist_tensor_spec}, attrs);
  EXPECT_EQ(infered_dist_attrs.first[0].dims_mapping(),
            std::vector<int64_t>({-1, -1}));
  EXPECT_EQ(infered_dist_attrs.first[1].dims_mapping(),
            std::vector<int64_t>({-1, 0}));
  EXPECT_EQ(infered_dist_attrs.second[0].dims_mapping(),
            std::vector<int64_t>({-1, 0}));
  EXPECT_EQ(infered_dist_attrs.second[0].is_partial(), false);
  VLOG(4) << "test2 done." << std::endl << std::endl << std::endl;

  // mk[1, 0],kn[-1,-1] --> mk[1, 0],kn[0, -1] = nm[1, -1] partial[0]: done
  x_dist_tensor_spec.set_dims_mapping({1, 0});
  y_dist_tensor_spec.set_dims_mapping({-1, -1});
  infered_dist_attrs = matmul_rule->InferForward(
      {x_dist_tensor_spec, y_dist_tensor_spec}, attrs);
  EXPECT_EQ(infered_dist_attrs.first[0].dims_mapping(),
            std::vector<int64_t>({1, 0}));
  EXPECT_EQ(infered_dist_attrs.first[1].dims_mapping(),
            std::vector<int64_t>({0, -1}));
  EXPECT_EQ(infered_dist_attrs.second[0].dims_mapping(),
            std::vector<int64_t>({1, -1}));
  EXPECT_EQ(infered_dist_attrs.second[0].is_partial(), true);
  EXPECT_EQ(infered_dist_attrs.second[0].partial_dims(),
            std::set<int64_t>({0}));
  VLOG(4) << "test3 done." << std::endl << std::endl << std::endl;

  // mk[-1,-1],kn[1,0] --> mk[-1, 1],kn[1, 0] = nm[-1, 0] partial[1]: done
  x_dist_tensor_spec.set_dims_mapping({-1, -1});
  y_dist_tensor_spec.set_dims_mapping({1, 0});
  infered_dist_attrs = matmul_rule->InferForward(
      {x_dist_tensor_spec, y_dist_tensor_spec}, attrs);
  EXPECT_EQ(infered_dist_attrs.first[0].dims_mapping(),
            std::vector<int64_t>({-1, 1}));
  EXPECT_EQ(infered_dist_attrs.first[1].dims_mapping(),
            std::vector<int64_t>({1, 0}));
  EXPECT_EQ(infered_dist_attrs.second[0].dims_mapping(),
            std::vector<int64_t>({-1, 0}));
  EXPECT_EQ(infered_dist_attrs.second[0].is_partial(), true);
  EXPECT_EQ(infered_dist_attrs.second[0].partial_dims(),
            std::set<int64_t>({1}));
  VLOG(4) << "test4 done." << std::endl << std::endl << std::endl;

  // abcmk[1, 0, -1, -1],kn[-1, -1] --> abcmk[1, 0, -1, -1],kn[-1, -1] =
  // abcmn[1, 0, -1, -1] partial[]: done
  x_dist_tensor_spec.set_shape({512, 48, 64, 32});
  x_dist_tensor_spec.set_dims_mapping({0, 1, -1, -1});
  y_dist_tensor_spec.set_dims_mapping({-1, -1});
  infered_dist_attrs = matmul_rule->InferForward(
      {x_dist_tensor_spec, y_dist_tensor_spec}, attrs);
  EXPECT_EQ(infered_dist_attrs.first[0].dims_mapping(),
            std::vector<int64_t>({0, 1, -1, -1}));
  EXPECT_EQ(infered_dist_attrs.first[1].dims_mapping(),
            std::vector<int64_t>({-1, -1}));
  EXPECT_EQ(infered_dist_attrs.second[0].dims_mapping(),
            std::vector<int64_t>({0, 1, -1, -1}));
  EXPECT_EQ(infered_dist_attrs.second[0].is_partial(), false);
  VLOG(4) << "test5 done." << std::endl << std::endl << std::endl;

  // abcmk[1, -1, -1, 0],kn[-1, -1] --> abcmk[1, -1, -1, 0],kn[0, -1] = abcmn[1,
  // -1, -1, -1] partial[0]: done
  x_dist_tensor_spec.set_dims_mapping({1, -1, -1, 0});
  y_dist_tensor_spec.set_dims_mapping({-1, -1});
  infered_dist_attrs = matmul_rule->InferForward(
      {x_dist_tensor_spec, y_dist_tensor_spec}, attrs);
  EXPECT_EQ(infered_dist_attrs.first[0].dims_mapping(),
            std::vector<int64_t>({1, -1, -1, 0}));
  EXPECT_EQ(infered_dist_attrs.first[1].dims_mapping(),
            std::vector<int64_t>({0, -1}));
  EXPECT_EQ(infered_dist_attrs.second[0].dims_mapping(),
            std::vector<int64_t>({1, -1, -1, -1}));
  EXPECT_EQ(infered_dist_attrs.second[0].is_partial(), true);
  EXPECT_EQ(infered_dist_attrs.second[0].partial_dims(),
            std::set<int64_t>({0}));
  VLOG(4) << "test6 done." << std::endl << std::endl << std::endl;

  // abcmk[1, -1, -1, 0], kn[-1, -1] --> abcmk[1, -1, -1, 0],kn[-1, -1] =
  // abcmn[1, -1, 0, -1] partial[]: done
  x_dist_tensor_spec.set_dims_mapping({1, -1, -1, 0});
  y_dist_tensor_spec.set_dims_mapping({-1, -1});
  attrs["trans_x"] = true;
  infered_dist_attrs = matmul_rule->InferForward(
      {x_dist_tensor_spec, y_dist_tensor_spec}, attrs);
  EXPECT_EQ(infered_dist_attrs.first[0].dims_mapping(),
            std::vector<int64_t>({1, -1, -1, 0}));
  EXPECT_EQ(infered_dist_attrs.first[1].dims_mapping(),
            std::vector<int64_t>({-1, -1}));
  EXPECT_EQ(infered_dist_attrs.second[0].dims_mapping(),
            std::vector<int64_t>({1, -1, 0, -1}));
  EXPECT_EQ(infered_dist_attrs.second[0].is_partial(), false);
  VLOG(4) << "test7 done." << std::endl << std::endl << std::endl;

  // abcmk[-1, -1, -1, -1], kn[1, 0] --> abcmk[-1, -1, -1, 0],kn[1, 0] =
  // abcmn[-1, -1, -1, 1] partial[0]: done
  x_dist_tensor_spec.set_dims_mapping({-1, -1, -1, -1});
  y_dist_tensor_spec.set_dims_mapping({1, 0});
  attrs["trans_x"] = false;
  attrs["trans_y"] = true;
  infered_dist_attrs = matmul_rule->InferForward(
      {x_dist_tensor_spec, y_dist_tensor_spec}, attrs);
  EXPECT_EQ(infered_dist_attrs.first[0].dims_mapping(),
            std::vector<int64_t>({-1, -1, -1, 0}));
  EXPECT_EQ(infered_dist_attrs.first[1].dims_mapping(),
            std::vector<int64_t>({1, 0}));
  EXPECT_EQ(infered_dist_attrs.second[0].dims_mapping(),
            std::vector<int64_t>({-1, -1, -1, 1}));
  EXPECT_EQ(infered_dist_attrs.second[0].is_partial(), true);
  EXPECT_EQ(infered_dist_attrs.second[0].partial_dims(),
            std::set<int64_t>({0}));
  infered_dist_attrs.second[0].clean_partial_dims(std::vector<int64_t>({0}));
  EXPECT_EQ(infered_dist_attrs.second[0].is_partial(), false);
  VLOG(4) << "test8 done." << std::endl << std::endl << std::endl;

  // abcmk[-1, -1, -1, -1], kn[1, 0] --> abcmk[-1, -1, -1, 0],kn[1, 0] =
  // abcmn[-1, -1, -1, 1] partial[0]: done
  x_dist_tensor_spec.set_dims_mapping({-1, -1, 0, 1});
  y_dist_tensor_spec.set_dims_mapping({1, 0});
  attrs["trans_y"] = true;
  attrs["trans_x"] = true;
  infered_dist_attrs = matmul_rule->InferForward(
      {x_dist_tensor_spec, y_dist_tensor_spec}, attrs);
  EXPECT_EQ(infered_dist_attrs.first[0].dims_mapping(),
            std::vector<int64_t>({-1, -1, 0, 1}));
  EXPECT_EQ(infered_dist_attrs.first[1].dims_mapping(),
            std::vector<int64_t>({-1, 0}));
  EXPECT_EQ(infered_dist_attrs.second[0].dims_mapping(),
            std::vector<int64_t>({-1, -1, 1, -1}));
  EXPECT_EQ(infered_dist_attrs.second[0].partial_dims(),
            std::set<int64_t>({0}));
  VLOG(4) << infered_dist_attrs.second[0].to_string();
  infered_dist_attrs.second[0].clean_partial_status();
  EXPECT_EQ(infered_dist_attrs.second[0].is_partial(), false);
  infered_dist_attrs.second[0].set_partial_status(std::vector<int64_t>({1}));
  EXPECT_EQ(infered_dist_attrs.second[0].verify_partial_status(), false);
  VLOG(4) << "test9 done." << std::endl << std::endl << std::endl;

  // abcmk[-1, -1, 1, 0], kn[1, 0] --> abcmk[-1, -1, -1, 0],kn[1, 0] =
  // abcmn[-1, -1, -1, 1] partial[0]: done
  x_dist_tensor_spec.set_dims_mapping({-1, -1, 1, 0});
  y_dist_tensor_spec.set_dims_mapping({1, 0});
  attrs["trans_y"] = true;
  attrs["trans_x"] = true;
  EXPECT_ANY_THROW(infered_dist_attrs = matmul_rule->InferForward(
                       {x_dist_tensor_spec, y_dist_tensor_spec}, attrs));
  // Error
  VLOG(4) << "test10 done." << std::endl << std::endl << std::endl;

  // abcmk[-1, -1, -1, -1], kn[1, 0] --> abcmk[-1, -1, -1, 0],kn[1, 0] =
  // abcmn[-1, -1, -1, 1] partial[0]:
  x_dist_tensor_spec.set_dims_mapping({-1, -1, 0, 1});
  y_dist_tensor_spec.set_dims_mapping({1, 0});
  attrs["trans_y"] = true;
  attrs["trans_x"] = true;
  infered_dist_attrs = matmul_rule->InferForward(
      {x_dist_tensor_spec, y_dist_tensor_spec}, attrs);
  EXPECT_ANY_THROW(infered_dist_attrs.second[0].clean_partial_dims(
      std::vector<int64_t>({1})));
  infered_dist_attrs.second[0].set_partial_status(std::vector<int64_t>({1}));
  EXPECT_EQ(infered_dist_attrs.second[0].is_partial(), true);
  EXPECT_EQ(infered_dist_attrs.second[0].partial_dims(),
            std::set<int64_t>({0, 1}));
  infered_dist_attrs.second[0].clean_partial_dims(std::vector<int64_t>({1}));
  EXPECT_EQ(infered_dist_attrs.second[0].partial_dims(),
            std::set<int64_t>({0}));
  infered_dist_attrs.second[0].clean_partial_dims(std::vector<int64_t>({0}));
  EXPECT_EQ(infered_dist_attrs.second[0].is_partial(), false);

  VLOG(4) << "test11 done." << std::endl << std::endl << std::endl;
}

TEST(LayerNormSPMDRule, Ctor) {
  // build input data class
  std::vector<int64_t> x_shape = {64, 32, 1024};
  std::vector<int64_t> scale_shape = {1024};
  std::vector<int64_t> bias_shape = {1024};

  std::vector<int64_t> mesh_shape = {2, 3};
  std::vector<int64_t> process_ids = {0, 1, 2, 3, 4, 5};
  std::vector<std::string> dim_names = {"x", "y"};
  ProcessMesh process_mesh(mesh_shape, process_ids, dim_names);

  TensorDistAttr x_dist_attr = TensorDistAttr();
  x_dist_attr.set_process_mesh(process_mesh);
  x_dist_attr.set_dims_mapping(std::vector<int64_t>({1, -1, -1}));
  x_dist_attr.set_dynamic_dims(std::vector<bool>({false, false, false}));

  TensorDistAttr scale_dist_attr = TensorDistAttr();
  scale_dist_attr.set_process_mesh(process_mesh);
  scale_dist_attr.set_dims_mapping(std::vector<int64_t>({-1}));
  scale_dist_attr.set_dynamic_dims(std::vector<bool>({false}));

  TensorDistAttr bias_dist_attr = TensorDistAttr();
  bias_dist_attr.set_process_mesh(process_mesh);
  bias_dist_attr.set_dims_mapping(std::vector<int64_t>({-1}));
  bias_dist_attr.set_dynamic_dims(std::vector<bool>({false}));

  DistTensorSpec x_dist_tensor_spec = DistTensorSpec(x_shape, x_dist_attr);
  DistTensorSpec scale_dist_tensor_spec =
      DistTensorSpec(scale_shape, scale_dist_attr);
  DistTensorSpec bias_dist_tensor_spec =
      DistTensorSpec(bias_shape, bias_dist_attr);

  paddle::framework::AttributeMap attrs;
  attrs["begin_norm_axis"] = 2;

  SPMDRuleBase* layer_norm_rule = SPMDRuleMap::Instance().Get("layer_norm");

  // ijk[1, -1, -1], k[-1], k[-1] --> ijk[1, -1, -1], z[1], z[1], z=ij,
  // begin_norm_axis=2
  std::pair<std::vector<TensorDistAttr>, std::vector<TensorDistAttr>>
      infered_dist_attrs = layer_norm_rule->InferForward(
          {x_dist_tensor_spec, scale_dist_tensor_spec, bias_dist_tensor_spec},
          attrs);

  size_t input_size = 3;
  size_t output_size = 3;
  EXPECT_EQ(infered_dist_attrs.first.size(), input_size);
  EXPECT_EQ(infered_dist_attrs.second.size(), output_size);

  EXPECT_EQ(infered_dist_attrs.first[0].dims_mapping(),
            std::vector<int64_t>({1, -1, -1}));
  EXPECT_EQ(infered_dist_attrs.first[1].dims_mapping(),
            std::vector<int64_t>({-1}));
  EXPECT_EQ(infered_dist_attrs.first[2].dims_mapping(),
            std::vector<int64_t>({-1}));
  EXPECT_EQ(infered_dist_attrs.second[0].dims_mapping(),
            std::vector<int64_t>({1, -1, -1}));
  EXPECT_EQ(infered_dist_attrs.second[1].dims_mapping(),
            std::vector<int64_t>({1}));
  EXPECT_EQ(infered_dist_attrs.second[2].dims_mapping(),
            std::vector<int64_t>({1}));
  VLOG(4) << "test1 done.";

  // ijk[1, 0, -1],k[0],k[0] --> error, begin_norm_axis=2
  x_dist_tensor_spec.set_dims_mapping({1, 0, -1});
  scale_dist_tensor_spec.set_dims_mapping({0});
  bias_dist_tensor_spec.set_dims_mapping({0});
  EXPECT_ANY_THROW(
      infered_dist_attrs = layer_norm_rule->InferForward(
          {x_dist_tensor_spec, scale_dist_tensor_spec, bias_dist_tensor_spec},
          attrs););
  VLOG(4) << "test2 done.";

  // ijk[0, -1, -1],y[-1],y[1] --> ijk[0, 1, -1], i[0], i[0], y=jk,
  // begin_norm_axis=1
  x_dist_tensor_spec.set_dims_mapping({0, -1, -1});
  scale_dist_tensor_spec.set_dims_mapping({-1});
  bias_dist_tensor_spec.set_dims_mapping({1});
  attrs["begin_norm_axis"] = 1;
  infered_dist_attrs = layer_norm_rule->InferForward(
      {x_dist_tensor_spec, scale_dist_tensor_spec, bias_dist_tensor_spec},
      attrs);
  EXPECT_EQ(infered_dist_attrs.first[0].dims_mapping(),
            std::vector<int64_t>({0, -1, -1}));
  EXPECT_EQ(infered_dist_attrs.first[1].dims_mapping(),
            std::vector<int64_t>({-1}));
  EXPECT_EQ(infered_dist_attrs.first[2].dims_mapping(),
            std::vector<int64_t>({-1}));
  EXPECT_EQ(infered_dist_attrs.second[0].dims_mapping(),
            std::vector<int64_t>({0, -1, -1}));
  EXPECT_EQ(infered_dist_attrs.second[1].dims_mapping(),
            std::vector<int64_t>({0}));
  EXPECT_EQ(infered_dist_attrs.second[2].dims_mapping(),
            std::vector<int64_t>({0}));
  VLOG(4) << "test2 done.";
}

}  // namespace auto_parallel
}  // namespace distributed
}  // namespace paddle
