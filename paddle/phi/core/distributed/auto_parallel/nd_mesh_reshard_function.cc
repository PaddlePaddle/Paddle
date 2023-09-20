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

#include "paddle/phi/core/distributed/auto_parallel/nd_mesh_reshard_function.h"

#include "glog/logging.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_tensor.h"
#include "paddle/phi/core/distributed/auto_parallel/p_to_r_reshard_function.h"
#include "paddle/phi/core/distributed/auto_parallel/r_to_p_reshard_function.h"
#include "paddle/phi/core/distributed/auto_parallel/r_to_s_reshard_function.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/s_to_r_reshard_function.h"

namespace phi {
namespace distributed {

namespace {
ProcessMesh GetSubProcessMesh(const ProcessMesh& mesh, int64_t axis) {
  int64_t shape_of_axis = mesh.dim_size(axis);
  std::vector<int64_t> shape = {shape_of_axis};
  std::vector<std::string> dim_names = {mesh.dim_names()[axis]};
  std::vector<int64_t> coord = GetCurRankCoordInMesh(mesh);

  std::vector<int64_t> process_ids;
  for (int64_t i = 0; i < shape_of_axis; ++i) {
    coord[axis] = i;
    int64_t rank = coord.back();
    for (int64_t j = coord.size() - 2; j >= 0; --j) {
      rank += coord[j] * mesh.dim_size(j + 1);
    }
    process_ids.emplace_back(rank);
  }

  ProcessMesh out_mesh(shape, process_ids, dim_names);
  return out_mesh;
}

// Given the input two dist_attr, traversing from high-dimension axis to
// low-dimension. Find and return the first different axis which is shard status
// between these two. For example, the input two dims_mapping are [-1, 0, -1,
// -1] and [-1, -1, 0, -1], the first diff shard axis is 2.
int64_t FindFirstDiffShardAxis(const TensorDistAttr& in_dist_attr,
                               const TensorDistAttr& out_dist_attr) {
  const auto& in_dims_mapping = in_dist_attr.dims_mapping();
  const auto& out_dims_mapping = out_dist_attr.dims_mapping();
  int64_t axis = -1;

  for (int64_t i = in_dims_mapping.size() - 1; i >= 0; --i) {
    if (in_dims_mapping[i] != out_dims_mapping[i]) {
      axis = i;
      break;
    }
  }

  return axis;
}

}  // namespace

bool SameNdMeshReshardFunction::IsSuitable(
    const DistTensor& in, const TensorDistAttr& out_dist_attr) {
  bool flag = true;

  flag &= (in.dist_attr().process_mesh() == out_dist_attr.process_mesh());
  flag &= (out_dist_attr.process_mesh().ndim() > 1);

  // check the input and output dims_mapping is not equal
  flag &= in.dist_attr() != out_dist_attr;

  return flag;
}

void SameNdMeshReshardFunction::Eval(phi::DeviceContext* dev_ctx,
                                     const DistTensor& in,
                                     const TensorDistAttr& out_dist_attr,
                                     DistTensor* out) {
  const auto& in_dist_attr = in.dist_attr();
  const auto& process_mesh = out_dist_attr.process_mesh();

  int64_t first_diff_axis = FindFirstDiffShardAxis(in_dist_attr, out_dist_attr);

  SetValue(out, in.value());
  SetDistProps(out, in.dims(), in_dist_attr);

  // 1. change all the partial status to replicated status if needed
  if (in_dist_attr.is_partial()) {
    const auto& in_partial_status = in_dist_attr.partial_status();
    const auto& out_partial_status = out_dist_attr.partial_status();
    for (const auto& kv : in_partial_status) {
      if (out_partial_status.count(kv.first) != 0) {
        continue;
      }
      VLOG(3) << "Step1: partial axis " << kv.first;
      // 1.1 Calculate the dist_attr after this transform
      TensorDistAttr real_out_dist_attr(out->dist_attr());
      real_out_dist_attr.clean_partial_dims({kv.first});

      // 1.2 Calculate the process_mesh on specific axis
      ProcessMesh sub_mesh = GetSubProcessMesh(process_mesh, kv.first);

      // 1.3 Calculate the input one dim dist attr
      TensorDistAttr in_one_dim_dist_attr(vectorize(in.dims()));
      in_one_dim_dist_attr.set_process_mesh(sub_mesh);
      in_one_dim_dist_attr.set_partial_status(std::vector<int64_t>{0});

      // 1.4 Calculate the output one dim dist attr
      TensorDistAttr out_one_dim_dist_attr(vectorize(in.dims()));
      out_one_dim_dist_attr.set_process_mesh(sub_mesh);

      // 1.5 Change from partial to replicated
      SetDistProps(out, in_one_dim_dist_attr);

      DistTensor tmp_result;
      PToRReshardFunction func;
      func.Eval(dev_ctx, *out, out_one_dim_dist_attr, &tmp_result);

      // 1.6 Reset to the right dist attr
      SetValue(out, tmp_result.value());
      SetDistProps(out, real_out_dist_attr);
    }
  }

  // 2. change all the shard status to replicated status
  for (int64_t i = first_diff_axis; i >= 0; --i) {
    int64_t in_mesh_axis = out->dist_attr().dims_mapping()[i];
    if (in_mesh_axis != -1) {
      VLOG(3) << "Step2: in_mesh axis " << in_mesh_axis;
      // 2.1 Calculate the dist_attr after this transform
      TensorDistAttr real_out_dist_attr(out->dist_attr());
      std::vector<int64_t> real_dims_mapping =
          real_out_dist_attr.dims_mapping();
      real_dims_mapping[i] = -1;
      real_out_dist_attr.set_dims_mapping(real_dims_mapping);

      // 2.2 Calculate the process_mesh on specific axis
      ProcessMesh sub_mesh = GetSubProcessMesh(process_mesh, in_mesh_axis);

      // 2.3 Calculate the input one dim dist attr
      TensorDistAttr in_one_dim_dist_attr(vectorize(in.dims()));
      in_one_dim_dist_attr.set_process_mesh(sub_mesh);
      std::vector<int64_t> in_one_dims_mapping =
          in_one_dim_dist_attr.dims_mapping();
      in_one_dims_mapping[i] = 0;
      in_one_dim_dist_attr.set_dims_mapping(in_one_dims_mapping);

      // 2.4 Calculate the output one dim dist attr
      TensorDistAttr out_one_dim_dist_attr(vectorize(in.dims()));
      out_one_dim_dist_attr.set_process_mesh(sub_mesh);

      // 2.5 Change from shard to replicated
      SetDistProps(out, in_one_dim_dist_attr);
      DistTensor tmp_result;
      SToRReshardFunction func;
      func.Eval(dev_ctx, *out, out_one_dim_dist_attr, &tmp_result);

      // 2.6 Reset to the right dist attr
      SetValue(out, tmp_result.value());
      SetDistProps(out, real_out_dist_attr);
    }
  }

  // 3. Change replicated to partial
  if (out_dist_attr.is_partial()) {
    const auto& in_partial_status = out->dist_attr().partial_status();
    const auto& out_partial_status = out_dist_attr.partial_status();
    for (const auto& kv : out_partial_status) {
      if (in_partial_status.count(kv.first) != 0) {
        continue;
      }
      VLOG(3) << "Step3: Partial status mesh axis " << kv.first;
      // 3.1 Calculate the dist_attr after this transform
      TensorDistAttr real_out_dist_attr(out->dist_attr());
      real_out_dist_attr.set_partial_status(std::vector<int64_t>{kv.first});

      // 3.2 Calculate the process_mesh on specific axis
      ProcessMesh sub_mesh = GetSubProcessMesh(process_mesh, kv.first);

      // 3.3 Calculate the input one dim dist attr
      TensorDistAttr in_one_dim_dist_attr(vectorize(in.dims()));
      in_one_dim_dist_attr.set_process_mesh(sub_mesh);

      // 3.4 Calculate the output one dim dist attr
      TensorDistAttr out_one_dim_dist_attr(vectorize(in.dims()));
      out_one_dim_dist_attr.set_process_mesh(sub_mesh);
      out_one_dim_dist_attr.set_partial_status(std::vector<int64_t>{0});

      // 3.5 Change from partial to replicated
      DistTensor tmp_result;
      SetDistProps(out, in_one_dim_dist_attr);
      RToPReshardFunction func;
      func.Eval(dev_ctx, *out, out_one_dim_dist_attr, &tmp_result);

      // 3.6 Reset to the right dist attr
      SetValue(out, tmp_result.value());
      SetDistProps(out, real_out_dist_attr);
    }
  }

  // 4. Change replicated to shard
  for (int64_t i = first_diff_axis; i >= 0; --i) {
    int64_t out_mesh_axis = out_dist_attr.dims_mapping()[i];
    if (out_mesh_axis != -1) {
      VLOG(3) << "Step4: out_mesh axis " << out_mesh_axis;
      // 4.1 Calculate the dist_attr after this transform
      TensorDistAttr real_out_dist_attr(out->dist_attr());
      std::vector<int64_t> real_dims_mapping =
          real_out_dist_attr.dims_mapping();
      real_dims_mapping[i] = out_mesh_axis;
      real_out_dist_attr.set_dims_mapping(real_dims_mapping);

      // 4.2 Calculate the process_mesh on specific axis
      ProcessMesh sub_mesh = GetSubProcessMesh(process_mesh, out_mesh_axis);

      // 4.3 Calculate the input one dim dist attr
      TensorDistAttr in_one_dim_dist_attr(vectorize(in.dims()));
      in_one_dim_dist_attr.set_process_mesh(sub_mesh);

      // 4.4 Calculate the output one dim dist attr
      TensorDistAttr out_one_dim_dist_attr(vectorize(in.dims()));
      out_one_dim_dist_attr.set_process_mesh(sub_mesh);
      std::vector<int64_t> out_one_dims_mapping =
          out_one_dim_dist_attr.dims_mapping();
      out_one_dims_mapping[i] = 0;
      out_one_dim_dist_attr.set_dims_mapping(out_one_dims_mapping);

      // 4.5 Change from replicated to shard
      DistTensor tmp_result;
      SetDistProps(out, in_one_dim_dist_attr);
      RToSReshardFunction func;
      func.Eval(dev_ctx, *out, out_one_dim_dist_attr, &tmp_result);

      // 4.6 Reset to the right dist attr
      SetValue(out, tmp_result.value());
      SetDistProps(out, real_out_dist_attr);
    }
  }
}

REGISTER_RESHARD_FUNC(SameNdMeshReshardFunction);

}  // namespace distributed
}  // namespace phi
