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
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_tensor.h"
#include "paddle/phi/core/distributed/auto_parallel/r_to_s_reshard_function.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard_function.h"
#include "paddle/phi/core/tensor_utils.h"

namespace phi {
namespace distributed {
namespace auto_parallel {
namespace tests {

std::shared_ptr<DistTensor> ConstructReplicatedDistCPU(
    phi::CPUContext* dev_ctx,
    const std::vector<int64_t>& shape,
    const ProcessMesh& mesh) {
  phi::CPUPlace cpu_place = dev_ctx->GetPlace();
  const DDim dims(shape.data(), shape.size());

  int64_t num_of_elems = 1;
  for (const auto& value : shape) {
    num_of_elems *= value;
  }

  phi::DenseTensor input_dense;
  float* input_dense_ptr = input_dense.mutable_data<float>(dims, cpu_place);

  std::vector<float> vec(num_of_elems);
  memcpy(input_dense_ptr, vec.data(), num_of_elems * sizeof(float));

  std::shared_ptr<TensorDistAttr> dist_attr =
      std::make_shared<TensorDistAttr>(shape);

  std::vector<int64_t> dims_mapping(shape.size(), -1);
  dist_attr->set_dims_mapping(dims_mapping);
  dist_attr->set_process_mesh(mesh);

  return std::make_shared<DistTensor>(
      std::make_shared<DenseTensor>(input_dense), dist_attr);
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
std::shared_ptr<DistTensor> ConstructReplicatedDistGPU(
    phi::GPUContext* dev_ctx,
    const std::vector<int64_t>& shape,
    const ProcessMesh& mesh) {
  phi::GPUPlace gpu_place = dev_ctx->GetPlace();
  phi::CPUPlace cpu_place;
  const DDim dims(shape.data(), shape.size());

  int64_t num_of_elems = 1;
  for (const auto& value : shape) {
    num_of_elems *= value;
  }

  phi::DenseTensor input_dense;
  phi::DenseTensor input_dense_gpu;
  float* input_dense_ptr = input_dense.mutable_data<float>(dims, cpu_place);

  std::vector<float> vec(num_of_elems);
  memcpy(input_dense_ptr, vec.data(), num_of_elems * sizeof(float));
  phi::Copy(*dev_ctx, input_dense, gpu_place, true, &input_dense_gpu);

  std::shared_ptr<TensorDistAttr> dist_attr =
      std::make_shared<TensorDistAttr>(shape);

  std::vector<int64_t> dims_mapping(shape.size(), -1);
  dist_attr->set_dims_mapping(dims_mapping);
  dist_attr->set_process_mesh(mesh);

  return std::make_shared<DistTensor>(
      std::make_shared<DenseTensor>(input_dense_gpu), dist_attr);
}
#endif

TEST(reshard_r_to_s, r_to_s_same_placement_cpu_1d_mesh) {
  setenv("PADDLE_TRAINER_ID", "1", 1);

  std::vector<int64_t> tensor_shape = {6, 8};
  phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
  auto* context = reinterpret_cast<phi::CPUContext*>(pool.Get(phi::CPUPlace()));

  std::vector<int64_t> mesh_shape = {4};
  std::vector<int64_t> process_ids = {0, 1, 2, 3};
  std::vector<std::string> dim_names = {"x"};
  ProcessMesh mesh(mesh_shape, process_ids, dim_names);

  std::shared_ptr<DistTensor> input =
      ConstructReplicatedDistCPU(context, tensor_shape, mesh);

  std::shared_ptr<TensorDistAttr> out_dist_attr =
      std::make_shared<TensorDistAttr>(tensor_shape);
  std::vector<int64_t> out_dims_mapping = {-1, 0};
  out_dist_attr->set_dims_mapping(out_dims_mapping);
  out_dist_attr->set_process_mesh(mesh);

  RToSReshardFunction r_to_s_func;
  std::shared_ptr<DistTensor> output =
      r_to_s_func.Eval(*context, *input, out_dist_attr);

  CHECK_EQ(r_to_s_func.IsSuitable(*input, out_dist_attr), true);
  CHECK_EQ(output->numel(), 12);
  CHECK_EQ(output->dims(), DDim({6, 2}));
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
TEST(reshard_r_to_s, r_to_s_same_placement_gpu_1d_mesh) {
  setenv("PADDLE_TRAINER_ID", "0", 0);

  std::vector<int64_t> tensor_shape = {6, 8, 4};
  phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
  auto* context = reinterpret_cast<phi::GPUContext*>(pool.Get(phi::GPUPlace()));

  std::vector<int64_t> mesh_shape = {6};
  std::vector<int64_t> process_ids = {0, 1, 2, 3, 4, 5};
  std::vector<std::string> dim_names = {"x"};
  ProcessMesh mesh(mesh_shape, process_ids, dim_names);

  std::shared_ptr<TensorDistAttr> out_dist_attr =
      std::make_shared<TensorDistAttr>(tensor_shape);
  std::vector<int64_t> out_dims_mapping = {0, -1};
  out_dist_attr->set_dims_mapping(out_dims_mapping);
  out_dist_attr->set_process_mesh(mesh);

  std::shared_ptr<DistTensor> input =
      ConstructReplicatedDistGPU(context, tensor_shape, mesh);

  RToSReshardFunction r_to_s_func;
  std::shared_ptr<DistTensor> output =
      r_to_s_func.Eval(*context, *input, out_dist_attr);

  CHECK_EQ(r_to_s_func.IsSuitable(*input, out_dist_attr), true);
  CHECK_EQ(output->numel(), 32);
  CHECK_EQ(output->dims(), DDim({1, 8, 4}));
}
#endif

TEST(reshard_r_to_s, r_to_s_diff_placement) {
  std::vector<int64_t> tensor_shape = {6, 8};
  phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
  auto* context = reinterpret_cast<phi::CPUContext*>(pool.Get(phi::CPUPlace()));

  std::vector<int64_t> mesh_shape = {4};
  std::vector<int64_t> process_ids = {0, 1, 2, 3};
  std::vector<std::string> dim_names = {"x"};
  ProcessMesh mesh(mesh_shape, process_ids, dim_names);

  std::shared_ptr<DistTensor> input =
      ConstructReplicatedDistCPU(context, tensor_shape, mesh);

  std::vector<int64_t> out_process_ids = {2, 3, 4, 5};
  ProcessMesh out_mesh(mesh_shape, out_process_ids, dim_names);
  std::shared_ptr<TensorDistAttr> out_dist_attr =
      std::make_shared<TensorDistAttr>(tensor_shape);
  std::vector<int64_t> out_dims_mapping = {-1, 0};
  out_dist_attr->set_dims_mapping(out_dims_mapping);
  out_dist_attr->set_process_mesh(out_mesh);

  RToSReshardFunction r_to_s_func;
  CHECK_EQ(r_to_s_func.IsSuitable(*input, out_dist_attr), false);
}

TEST(reshard_r_to_s, r_to_s_same_placement_nd_mesh) {
  std::vector<int64_t> tensor_shape = {6, 12};
  phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
  auto* context = reinterpret_cast<phi::CPUContext*>(pool.Get(phi::CPUPlace()));

  std::vector<int64_t> mesh_shape = {4, 2};
  std::vector<int64_t> process_ids = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<std::string> dim_names = {"x", "y"};
  ProcessMesh mesh(mesh_shape, process_ids, dim_names);

  std::shared_ptr<DistTensor> input =
      ConstructReplicatedDistCPU(context, tensor_shape, mesh);

  std::shared_ptr<TensorDistAttr> out_dist_attr =
      std::make_shared<TensorDistAttr>(tensor_shape);
  std::vector<int64_t> out_dims_mapping = {1, 0};
  out_dist_attr->set_dims_mapping(out_dims_mapping);
  out_dist_attr->set_process_mesh(mesh);

  RToSReshardFunction r_to_s_func;

  CHECK_EQ(r_to_s_func.IsSuitable(*input, out_dist_attr), false);
}

}  // namespace tests
}  // namespace auto_parallel
}  // namespace distributed
}  // namespace phi
