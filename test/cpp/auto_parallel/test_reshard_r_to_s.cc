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

#include <cstdlib>
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_tensor.h"
#include "paddle/phi/core/distributed/auto_parallel/r_to_s_reshard_function.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard_function.h"
#include "paddle/phi/core/kernel_factory.h"
#include "test/cpp/phi/core/allocator.h"

namespace phi {
namespace distributed {
namespace auto_parallel {
namespace tests {

std::shared_ptr<DistTensor> ConstructReplicatedDistTensor(
    const std::vector<int64_t>& shape,
    const DataLayout& layout,
    const DataType& dtype,
    const ProcessMesh& mesh) {
  const DDim dims(shape.data(), shape.size());
  const LoD lod{};
  DenseTensorMeta meta(dtype, dims, layout, lod);

  auto fancy_allocator =
      std::unique_ptr<Allocator>(new phi::tests::FancyAllocator);
  auto* alloc = fancy_allocator.get();
  std::shared_ptr<TensorDistAttr> dist_attr =
      std::make_shared<TensorDistAttr>(shape);

  std::vector<int64_t> dims_mapping(shape.size(), -1);
  dist_attr->set_dims_mapping(dims_mapping);
  dist_attr->set_process_mesh(mesh);

  return std::make_shared<DistTensor>(alloc, meta, dist_attr);
}

TEST(reshard_r_to_s, r_to_s_same_placement_1d_mesh) {
  setenv("PADDLE_TRAINER_ID", "1", 1);

  std::vector<int64_t> tensor_shape = {6, 12};
  const DataType dtype{DataType::FLOAT32};
  const DataLayout layout{DataLayout::NHWC};

  std::vector<int64_t> mesh_shape = {4};
  std::vector<int64_t> process_ids = {0, 1, 2, 3};
  std::vector<std::string> dim_names = {"x"};
  ProcessMesh mesh(mesh_shape, process_ids, dim_names);

  std::shared_ptr<DistTensor> input =
      ConstructReplicatedDistTensor(tensor_shape, layout, dtype, mesh);
  int64_t split_axis = 1;

  // Use process mesh axis 0 to split tensor axis 1
  std::shared_ptr<TensorDistAttr> out_dist_attr =
      std::make_shared<TensorDistAttr>(tensor_shape);
  std::vector<int64_t> out_dims_mapping(tensor_shape.size(), -1);
  out_dims_mapping[split_axis] = 0;
  out_dist_attr->set_dims_mapping(out_dims_mapping);
  out_dist_attr->set_process_mesh(mesh);

  RToSReshardFunction r_to_s_func;
  KernelKey kernel_key = {Backend::CPU, layout, dtype};
  std::shared_ptr<DistTensor> output =
      r_to_s_func.Eval(kernel_key, *input, out_dist_attr);

  CHECK_EQ(r_to_s_func.Check(*input, out_dist_attr), true);
  CHECK_EQ(output->numel(), 18);
  CHECK_EQ(output->dims(), DDim({6, 3}));
}

TEST(reshard_r_to_s, r_to_s_diff_placement) {
  std::vector<int64_t> tensor_shape = {6, 12};
  const DataType dtype{DataType::FLOAT32};
  const DataLayout layout{DataLayout::NHWC};

  std::vector<int64_t> mesh_shape = {4};
  std::vector<int64_t> process_ids = {0, 1, 2, 3};
  std::vector<std::string> dim_names = {"x"};
  ProcessMesh mesh(mesh_shape, process_ids, dim_names);

  std::shared_ptr<DistTensor> input =
      ConstructReplicatedDistTensor(tensor_shape, layout, dtype, mesh);
  int64_t split_axis = 1;

  std::vector<int64_t> out_process_ids = {2, 3, 4, 5};
  ProcessMesh out_mesh(mesh_shape, out_process_ids, dim_names);
  std::shared_ptr<TensorDistAttr> out_dist_attr =
      std::make_shared<TensorDistAttr>(tensor_shape);
  std::vector<int64_t> out_dims_mapping(tensor_shape.size(), -1);
  out_dims_mapping[split_axis] = 0;
  out_dist_attr->set_dims_mapping(out_dims_mapping);
  out_dist_attr->set_process_mesh(out_mesh);

  RToSReshardFunction r_to_s_func;
  CHECK_EQ(r_to_s_func.Check(*input, out_dist_attr), false);
}

TEST(reshard_r_to_s, r_to_s_same_placement_nd_mesh) {
  setenv("PADDLE_TRAINER_ID", "6", 1);

  std::vector<int64_t> tensor_shape = {6, 12};
  const DataType dtype{DataType::FLOAT32};
  const DataLayout layout{DataLayout::NHWC};

  std::vector<int64_t> mesh_shape = {4, 2};
  std::vector<int64_t> process_ids = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<std::string> dim_names = {"x", "y"};
  ProcessMesh mesh(mesh_shape, process_ids, dim_names);

  std::shared_ptr<DistTensor> input =
      ConstructReplicatedDistTensor(tensor_shape, layout, dtype, mesh);

  // Use process mesh axis 0 to split tensor axis 1, use process mesh axis 1 to
  // split tensor axis 0
  std::shared_ptr<TensorDistAttr> out_dist_attr =
      std::make_shared<TensorDistAttr>(tensor_shape);
  std::vector<int64_t> out_dims_mapping = {1, 0};
  out_dist_attr->set_dims_mapping(out_dims_mapping);
  out_dist_attr->set_process_mesh(mesh);

  RToSReshardFunction r_to_s_func;
  KernelKey kernel_key = {Backend::CPU, layout, dtype};
  std::shared_ptr<DistTensor> output =
      r_to_s_func.Eval(kernel_key, *input, out_dist_attr);

  CHECK_EQ(r_to_s_func.Check(*input, out_dist_attr), true);
  CHECK_EQ(output->numel(), 9);
  CHECK_EQ(output->dims(), DDim({3, 3}));
}

}  // namespace tests
}  // namespace auto_parallel
}  // namespace distributed
}  // namespace phi
