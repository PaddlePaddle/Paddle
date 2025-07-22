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

#include "paddle/phi/core/distributed/auto_parallel/reshard/nd_mesh_reshard_function.h"

#include "glog/logging.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_tensor.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/p_to_r_reshard_function.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/p_to_s_reshard_function.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/r_to_p_reshard_function.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/r_to_s_reshard_function.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/reshard_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/s_to_r_reshard_function.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/same_status_reshard_function.h"
#include "paddle/phi/core/distributed/store/store_utils.h"

namespace phi::distributed {

namespace {
ProcessMesh GetSubProcessMesh(const ProcessMesh& mesh, int64_t axis) {
  int64_t shape_of_axis = mesh.dim_size(axis);
  std::vector<int64_t> shape = {shape_of_axis};
  std::vector<std::string> dim_names = {mesh.dim_names()[axis]};
  std::vector<int64_t> coord = GetCurRankCoordInMesh(mesh);

  std::vector<int64_t> process_ids;
  for (int64_t i = 0; i < shape_of_axis; ++i) {
    coord[axis] = i;
    int64_t rank = 0;
    int64_t degree = 1;
    for (int64_t j = static_cast<int64_t>(coord.size() - 1); j >= 0; --j) {
      rank += coord[j] * degree;
      degree *= mesh.dim_size(j);
    }
    process_ids.emplace_back(mesh.process_ids()[rank]);
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
  const auto& in_dims_mapping = in_dist_attr.multi_dims_mapping();
  const auto& out_dims_mapping = out_dist_attr.multi_dims_mapping();

  VLOG(3) << "In find diff axis, in dim mapping "
          << auto_parallel::str_join(in_dims_mapping) << ", out dim mapping "
          << auto_parallel::str_join(out_dims_mapping);
  int64_t axis = -1;

  for (int64_t i = static_cast<int64_t>(in_dims_mapping.size() - 1); i >= 0;
       --i) {
    if (in_dims_mapping[i] != out_dims_mapping[i]) {
      axis = i;
      break;
    }
    auto predicate = [&in_dist_attr, &out_dist_attr](int64_t mesh_dim) {
      if (in_dist_attr.get_split_factor(mesh_dim) !=
          out_dist_attr.get_split_factor(mesh_dim)) {
        return true;
      }
      return false;
    };
    if (std::any_of(
            in_dims_mapping[i].begin(), in_dims_mapping[i].end(), predicate)) {
      axis = i;
      break;
    }
  }

  return axis;
}
}  // namespace

class ReshardContext final {
 public:
  ReshardContext(phi::DeviceContext* dev_ctx,
                 const DistTensor& in,
                 const TensorDistAttr& out_dist_attr,
                 DistTensor* out)
      : dev_ctx(dev_ctx), in(in), out_dist_attr(out_dist_attr), out(out) {}

  TensorDistAttr CreateOneDimDistAttr(
      const ProcessMesh& sub_mesh,
      bool is_partial = false,
      std::optional<ReduceType> reduce_type = std::nullopt,
      std::optional<int64_t> cur_tensor_dim = std::nullopt,
      std::optional<int64_t> cur_mesh_split_factor = std::nullopt) const {
    TensorDistAttr dist_attr(common::vectorize(in.dims()));
    dist_attr.set_process_mesh(sub_mesh);

    if (is_partial) {
      if (reduce_type) {
        dist_attr.set_partial_status(std::vector<int64_t>{0},
                                     reduce_type.value());
      } else {
        dist_attr.set_partial_status(std::vector<int64_t>{0});
      }
    }

    if (cur_tensor_dim) {
      auto dims_mapping = dist_attr.multi_dims_mapping();
      PADDLE_ENFORCE_GE(
          cur_tensor_dim.value(),
          0,
          common::errors::InvalidArgument(
              "tensor dim should be greater than or equal to 0, but got %d.",
              cur_tensor_dim.value()));
      dims_mapping[cur_tensor_dim.value()] = {0};
      dist_attr.set_dims_mapping(dims_mapping);
      if (cur_mesh_split_factor) {
        dist_attr.set_split_factor(0, cur_mesh_split_factor.value());
      }
    }
    return dist_attr;
  }

  ProcessMesh GetSubProcessMesh(int64_t axis) const {
    return phi::distributed::GetSubProcessMesh(out_dist_attr.process_mesh(),
                                               axis);
  }

  phi::DeviceContext* dev_ctx;
  const DistTensor& in;
  const TensorDistAttr& out_dist_attr;
  DistTensor* out;
  DistTensor tmp_result;
};

template <typename ReshardFunc>
class SingleDimReshardStrategy
    : public SameNdMeshReshardFunction::ReshardStrategy {
 public:
  SingleDimReshardStrategy(int64_t cur_tensor_dim,
                           int64_t cur_mesh_dim,
                           ReshardContext ctx)
      : cur_tensor_dim_(cur_tensor_dim),
        cur_mesh_dim_(cur_mesh_dim),
        ctx_(ctx) {}
  void Eval() override {
    auto cur_dist_attr = CalculateNewDistAttr();
    VLOG(3) << "New Dist Attr " << cur_dist_attr;
    auto sub_mesh = ctx_.GetSubProcessMesh(cur_mesh_dim_);
    VLOG(3) << "Get Sub Mesh " << sub_mesh;
    auto in_one_dim = CreateOneDimInDistAttr(sub_mesh);
    VLOG(3) << "One dim In Attr " << in_one_dim;
    auto out_one_dim = CreateOneDimOutDistAttr(sub_mesh);
    VLOG(3) << "One dim Out Attr " << out_one_dim;

    SetDistProps(ctx_.out, in_one_dim);
    VLOG(3) << "Set One dim In Attr";
    ReshardFunc func;
    func.Eval(ctx_.dev_ctx, *ctx_.out, out_one_dim, &ctx_.tmp_result);
    VLOG(3) << "Finish reshard func.";
    SetValue(ctx_.out, ctx_.tmp_result.value());
    VLOG(3) << "Set local value";
    SetDistProps(ctx_.out, cur_dist_attr);
    VLOG(3) << "Set Cur Dist Attr";
  }

 private:
  virtual TensorDistAttr CalculateNewDistAttr() const = 0;
  virtual TensorDistAttr CreateOneDimInDistAttr(
      const ProcessMesh& sub_mesh) const = 0;
  virtual TensorDistAttr CreateOneDimOutDistAttr(
      const ProcessMesh& sub_mesh) const = 0;

 protected:
  int64_t cur_tensor_dim_;
  int64_t cur_mesh_dim_;
  ReshardContext ctx_;
};

class PartialToReplicate final
    : public SingleDimReshardStrategy<PToRReshardFunction> {
 private:
  using SingleDimReshardStrategy<PToRReshardFunction>::SingleDimReshardStrategy;

  TensorDistAttr CalculateNewDistAttr() const override {
    auto real_out_attr = ctx_.out->dist_attr();
    real_out_attr.clean_partial_dims({cur_mesh_dim_});
    return real_out_attr;
  }

  TensorDistAttr CreateOneDimInDistAttr(
      const ProcessMesh& sub_mesh) const override {
    auto input_reduce_type =
        ctx_.out->dist_attr().partial_status().at(cur_mesh_dim_);
    return ctx_.CreateOneDimDistAttr(sub_mesh, true, input_reduce_type);
  }

  TensorDistAttr CreateOneDimOutDistAttr(
      const ProcessMesh& sub_mesh) const override {
    return ctx_.CreateOneDimDistAttr(sub_mesh);
  }
};

class ShardToReplicate final
    : public SingleDimReshardStrategy<SToRReshardFunction> {
 private:
  using SingleDimReshardStrategy<SToRReshardFunction>::SingleDimReshardStrategy;

  TensorDistAttr CalculateNewDistAttr() const override {
    auto real_out_attr = ctx_.out->dist_attr();
    std::vector<std::vector<int64_t>> real_dims_mapping =
        real_out_attr.multi_dims_mapping();
    real_dims_mapping[cur_tensor_dim_] = {};
    real_out_attr.set_dims_mapping(real_dims_mapping);
    real_out_attr.clear_split_factor(cur_mesh_dim_);
    return real_out_attr;
  }

  TensorDistAttr CreateOneDimInDistAttr(
      const ProcessMesh& sub_mesh) const override {
    auto split_factor = ctx_.out->dist_attr().get_split_factor(cur_mesh_dim_);
    VLOG(3) << "In S To R, cur mesh dim is " << cur_mesh_dim_
            << ", split factor is " << split_factor;
    return ctx_.CreateOneDimDistAttr(
        sub_mesh, false, std::nullopt, cur_tensor_dim_, split_factor);
  }

  TensorDistAttr CreateOneDimOutDistAttr(
      const ProcessMesh& sub_mesh) const override {
    return ctx_.CreateOneDimDistAttr(sub_mesh);
  }
};

class ReplicateToPartial final
    : public SingleDimReshardStrategy<RToPReshardFunction> {
 private:
  using SingleDimReshardStrategy<RToPReshardFunction>::SingleDimReshardStrategy;
  TensorDistAttr CalculateNewDistAttr() const override {
    TensorDistAttr real_out_dist_attr = ctx_.out->dist_attr();
    real_out_dist_attr.set_partial_status(std::vector<int64_t>{cur_mesh_dim_});
    return real_out_dist_attr;
  }

  TensorDistAttr CreateOneDimInDistAttr(
      const ProcessMesh& sub_mesh) const override {
    return ctx_.CreateOneDimDistAttr(sub_mesh);
  }

  TensorDistAttr CreateOneDimOutDistAttr(
      const ProcessMesh& sub_mesh) const override {
    return ctx_.CreateOneDimDistAttr(sub_mesh, true);
  }
};

class ReplicateToShard final
    : public SingleDimReshardStrategy<RToSReshardFunction> {
 private:
  using SingleDimReshardStrategy<RToSReshardFunction>::SingleDimReshardStrategy;

  TensorDistAttr CalculateNewDistAttr() const override {
    TensorDistAttr real_out_dist_attr(ctx_.out->dist_attr());
    std::vector<std::vector<int64_t>> real_dims_mapping =
        real_out_dist_attr.multi_dims_mapping();
    real_dims_mapping[cur_tensor_dim_].push_back(cur_mesh_dim_);
    real_out_dist_attr.set_dims_mapping(real_dims_mapping);

    auto split_factor = ctx_.out_dist_attr.get_split_factor(cur_mesh_dim_);
    real_out_dist_attr.set_split_factor(cur_mesh_dim_, split_factor);
    return real_out_dist_attr;
  }

  TensorDistAttr CreateOneDimInDistAttr(
      const ProcessMesh& sub_mesh) const override {
    return ctx_.CreateOneDimDistAttr(sub_mesh);
  }

  TensorDistAttr CreateOneDimOutDistAttr(
      const ProcessMesh& sub_mesh) const override {
    auto split_factor = ctx_.out_dist_attr.get_split_factor(cur_mesh_dim_);
    VLOG(3) << "In R to S mesh dim is " << cur_mesh_dim_ << ", split factor is "
            << split_factor;
    return ctx_.CreateOneDimDistAttr(
        sub_mesh, false, std::nullopt, cur_tensor_dim_, split_factor);
  }
};

class PartialToShard final
    : public SingleDimReshardStrategy<PToSReshardFunction> {
 private:
  using SingleDimReshardStrategy<PToSReshardFunction>::SingleDimReshardStrategy;

  TensorDistAttr CalculateNewDistAttr() const override {
    TensorDistAttr real_out_dist_attr(ctx_.out->dist_attr());
    std::vector<std::vector<int64_t>> real_dims_mapping =
        real_out_dist_attr.multi_dims_mapping();
    real_dims_mapping[cur_tensor_dim_].push_back(cur_mesh_dim_);
    real_out_dist_attr.set_dims_mapping(real_dims_mapping);
    if (real_out_dist_attr.is_partial(cur_mesh_dim_)) {
      real_out_dist_attr.clean_partial_dims({cur_mesh_dim_});
    }

    auto split_factor = ctx_.out_dist_attr.get_split_factor(cur_mesh_dim_);
    real_out_dist_attr.set_split_factor(cur_mesh_dim_, split_factor);
    return real_out_dist_attr;
  }

  TensorDistAttr CreateOneDimInDistAttr(
      const ProcessMesh& sub_mesh) const override {
    auto input_reduce_type =
        ctx_.out->dist_attr().partial_status().at(cur_mesh_dim_);
    return ctx_.CreateOneDimDistAttr(sub_mesh, true, input_reduce_type);
  }

  TensorDistAttr CreateOneDimOutDistAttr(
      const ProcessMesh& sub_mesh) const override {
    auto split_factor = ctx_.out_dist_attr.get_split_factor(cur_mesh_dim_);
    VLOG(3) << "In P to S mesh dim is " << cur_mesh_dim_ << ", split factor is "
            << split_factor;
    return ctx_.CreateOneDimDistAttr(
        sub_mesh, false, std::nullopt, cur_tensor_dim_, split_factor);
  }
};

void ProcessPartialToReplicated(phi::DeviceContext* dev_ctx,
                                const DistTensor& in,
                                TensorDistAttr out_dist_attr,
                                DistTensor* out) {
  if (in.dist_attr().is_partial()) {
    auto partial_status = in.dist_attr().partial_status();
    auto out_partial_status = out_dist_attr.partial_status();
    ReshardContext ctx(dev_ctx, in, out_dist_attr, out);
    for (const auto& [k, v] : partial_status) {
      VLOG(3) << "Step1: partial axis " << k;
      if (out_partial_status.count(k) != 0 || out_dist_attr.is_shard(k)) {
        continue;
      }
      auto strategy = std::make_unique<PartialToReplicate>(-1, k, ctx);
      strategy->Eval();
    }
    VLOG(3) << "After P to R, dist attr is " << out->dist_attr();
  }
}

void ProcessShardToReplicated(phi::DeviceContext* dev_ctx,
                              const DistTensor& in,
                              TensorDistAttr out_dist_attr,
                              DistTensor* out) {
  auto is_same_shard = [&out_dist_attr, &out](
                           std::vector<int64_t> in_mesh_axis,
                           std::vector<int64_t> out_mesh_axis) {
    if (in_mesh_axis != out_mesh_axis) {
      return false;
    }
    for (auto dim : in_mesh_axis) {
      if (out_dist_attr.get_split_factor(dim) !=
          out->dist_attr().get_split_factor(dim)) {
        return false;
      }
    }
    return true;
  };
  int64_t first_diff_axis =
      FindFirstDiffShardAxis(out->dist_attr(), out_dist_attr);
  VLOG(3) << "In S to R, fist diff axis is " << first_diff_axis;
  for (int cur_tensor_dim = first_diff_axis; cur_tensor_dim >= 0;
       --cur_tensor_dim) {
    auto in_mesh_axis = out->dist_attr().multi_dims_mapping()[cur_tensor_dim];
    auto out_mesh_axis = out_dist_attr.multi_dims_mapping()[cur_tensor_dim];
    if (in_mesh_axis.size() == 0 ||
        is_same_shard(in_mesh_axis, out_mesh_axis)) {
      continue;
    }
    VLOG(3) << "Step2: in_mesh axis " << auto_parallel::str_join(in_mesh_axis);
    ReshardContext ctx(dev_ctx, in, out_dist_attr, out);
    for (int64_t idx = in_mesh_axis.size() - 1; idx >= 0; idx--) {
      int64_t cur_mesh_dim = in_mesh_axis.at(idx);
      auto strategy =
          std::make_unique<ShardToReplicate>(cur_tensor_dim, cur_mesh_dim, ctx);
      strategy->Eval();
    }
  }
  if (first_diff_axis >= 0) {
    VLOG(3) << "After S to R, dist attr is " << out->dist_attr();
  }
}

void ProcessReplicatedToPartial(phi::DeviceContext* dev_ctx,
                                const DistTensor& in,
                                TensorDistAttr out_dist_attr,
                                DistTensor* out) {
  if (out_dist_attr.is_partial()) {
    const auto& in_partial_status = out->dist_attr().partial_status();
    const auto& out_partial_status = out_dist_attr.partial_status();
    for (const auto& [k, v] : out_partial_status) {
      if (in_partial_status.count(k) != 0) {
        continue;
      }
      VLOG(3) << "Step3: Partial status mesh axis " << k;
      ReshardContext ctx(dev_ctx, in, out_dist_attr, out);
      auto strategy = std::make_unique<ReplicateToPartial>(-1, k, ctx);
      strategy->Eval();
    }
    VLOG(3) << "After R to P, dist attr is " << out->dist_attr();
  }
}

void ProcessReplicateOrPartialToShard(phi::DeviceContext* dev_ctx,
                                      const DistTensor& in,
                                      TensorDistAttr out_dist_attr,
                                      DistTensor* out) {
  int64_t first_diff_axis =
      FindFirstDiffShardAxis(out->dist_attr(), out_dist_attr);
  VLOG(3) << "In P or R to S, fist diff axis is " << first_diff_axis;
  for (int64_t cur_tensor_dim = first_diff_axis; cur_tensor_dim >= 0;
       --cur_tensor_dim) {
    const auto& in_mesh_axis =
        out->dist_attr().multi_dims_mapping()[cur_tensor_dim];
    const auto& out_mesh_axis =
        out_dist_attr.multi_dims_mapping()[cur_tensor_dim];
    if (in_mesh_axis == out_mesh_axis) {
      continue;
    }

    const auto& in_partial_status = out->dist_attr().partial_status();
    ReshardContext ctx(dev_ctx, in, out_dist_attr, out);

    for (auto cur_mesh_dim : out_mesh_axis) {
      bool is_partial = in_partial_status.count(cur_mesh_dim) != 0;
      VLOG(3) << "Step4: out_mesh axis : " << cur_mesh_dim
              << "; partial state :" << is_partial;
      std::shared_ptr<SameNdMeshReshardFunction::ReshardStrategy> strategy;
      if (is_partial) {
        strategy =
            std::make_unique<PartialToShard>(cur_tensor_dim, cur_mesh_dim, ctx);
      } else {
        strategy = std::make_unique<ReplicateToShard>(
            cur_tensor_dim, cur_mesh_dim, ctx);
      }
      strategy->Eval();
    }
  }
  if (first_diff_axis >= 0) {
    VLOG(3) << "After P or R to S, dist attr is " << out->dist_attr();
  }
}

bool SameNdMeshReshardFunction::IsSuitable(
    const DistTensor& in, const TensorDistAttr& out_dist_attr) {
  RESHARD_SHORTCUT_IF_FALSE(in.dist_attr().process_mesh() ==
                            out_dist_attr.process_mesh());
  RESHARD_SHORTCUT_IF_FALSE(out_dist_attr.process_mesh().ndim() > 1);

  // check the input and output dims_mapping is not equal
  RESHARD_SHORTCUT_IF_FALSE(in.dist_attr() != out_dist_attr);

  return true;
}

void SameNdMeshReshardFunction::Eval(phi::DeviceContext* dev_ctx,
                                     const DistTensor& in,
                                     const TensorDistAttr& out_dist_attr,
                                     DistTensor* out) {
  auto out_dist_attr_orig = out_dist_attr;
  SetValue(out, in.value());
  SetDistProps(out, in.dims(), in.dist_attr());
  // 1. change all the partial status to replicated status if needed
  ProcessPartialToReplicated(dev_ctx, in, out_dist_attr_orig, out);
  // 2. change all the shard status to replicated status
  ProcessShardToReplicated(dev_ctx, in, out_dist_attr_orig, out);
  // 3. Change replicated to partial
  ProcessReplicatedToPartial(dev_ctx, in, out_dist_attr_orig, out);
  // 4. Change replicated/partial to shard
  ProcessReplicateOrPartialToShard(dev_ctx, in, out_dist_attr_orig, out);

  // TODO(lfw): refine this, now is reports wrong info.
  // Final attr check
  // PADDLE_ENFORCE_EQ(out->dist_attr() == out_dist_attr_orig,
  //                   true,
  //                   ::common::errors::InvalidArgument("Expected that out of
  //                   reshard has to be target dist, " "out dist att is " +
  //                   out->dist_attr().to_string() + ", but target is " +
  //                   out_dist_attr_orig.to_string()));
}

bool CrossNdMeshReshardFunction::IsSuitable(
    const DistTensor& in, const TensorDistAttr& out_dist_attr) {
  const ProcessMesh& in_process_mesh = in.dist_attr().process_mesh();
  const ProcessMesh& out_process_mesh = out_dist_attr.process_mesh();
  RESHARD_SHORTCUT_IF_FALSE(in_process_mesh != out_process_mesh);
  RESHARD_SHORTCUT_IF_FALSE(in_process_mesh.shape() ==
                            out_process_mesh.shape());
  RESHARD_SHORTCUT_IF_FALSE(out_dist_attr.process_mesh().ndim() > 1);

  // check the input and output dims_mapping is not equal
  RESHARD_SHORTCUT_IF_FALSE(in.dist_attr() != out_dist_attr);

  return true;
}

void CrossNdMeshReshardFunction::Eval(DeviceContext* dev_ctx,
                                      const DistTensor& in,
                                      const TensorDistAttr& out_dist_attr,
                                      DistTensor* out) {
  VLOG(3) << "Call " << Name();
  const auto& in_dist_attr = in.dist_attr();

  // Construct a `DistTensor` by `dtype` of `in` tensor to avoid using default
  // dtype `float32`. The default dtype `float32` may cause error in amp.
  DistTensor tmp_result(in.dtype());
  TensorDistAttr in_dist_attr_shard = in_dist_attr;
  in_dist_attr_shard.set_partial_status(out_dist_attr.partial_status());
  in_dist_attr_shard.set_dims_mapping(out_dist_attr.dims_mapping());

  int64_t cur_global_rank = GetCurGlobalRank();
  if (in_dist_attr.process_mesh().contains(cur_global_rank)) {
    SameNdMeshReshardFunction same_nd_reshard_func;
    PADDLE_ENFORCE(
        same_nd_reshard_func.IsSuitable(in, in_dist_attr_shard),
        common::errors::InvalidArgument(
            "Invoke the same nd reshard function is not valid from %s to %s.",
            in_dist_attr,
            in_dist_attr_shard));
    same_nd_reshard_func.Eval(dev_ctx, in, in_dist_attr_shard, &tmp_result);
  } else {
    SetDistProps(&tmp_result, in.dims(), in_dist_attr_shard);
  }

  SameStatusReshardFunction same_status_func;
  PADDLE_ENFORCE(
      same_status_func.IsSuitable(tmp_result, out_dist_attr),
      common::errors::InvalidArgument("Invoke the same status reshard function "
                                      "is not valid from %s to %s.",
                                      tmp_result.dist_attr(),
                                      out_dist_attr));
  same_status_func.Eval(dev_ctx, tmp_result, out_dist_attr, out);
}

}  // namespace phi::distributed
