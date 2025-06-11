#include "paddle/phi/infermeta/spmd_rules/index_put.h"

#include "glog/logging.h"

#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi::distributed {

using phi::distributed::auto_parallel::str_join;

SpmdInfo IndexPutInferSpmd(const DistMetaTensor& x,
                           const std::vector<DistMetaTensor>& indices,
                           const DistMetaTensor& value,
                           bool accumulate) {
  // Step0: Verify Input Args Based on IndexPut Logic
  auto x_shape = phi::vectorize(x.dims());
  int x_ndim = x_shape.size();
  TensorDistAttr x_dist_attr_src = x.dist_attr();
  std::vector<int64_t> x_dims_mapping = x_dist_attr_src.dims_mapping();
  PADDLE_ENFORCE_EQ(
      x_ndim,
      x_dims_mapping.size(),
      phi::errors::InvalidArgument("The Tensor X's rank [%d] and X's "
                                   "dims_mapping size [%d] are not matched.",
                                   x_ndim,
                                   x_dims_mapping.size()));

  std::vector<std::vector<int64_t>> indices_shape;
  std::vector<std::vector<int64_t>> indices_dims_mapping;
  std::vector<TensorDistAttr> indices_dist_attr_src;
  for (int i = 0; i < x_ndim; ++i) {
    auto index_shape = phi::vectorize(indices[i].dims());
    int index_ndim = index_shape.size();
    TensorDistAttr index_dist_attr_src = indices[i].dist_attr();
    std::vector<int64_t> index_dims_mapping =
        index_dist_attr_src.dims_mapping();
    indices_shape.emplace_back(index_shape);
    indices_dims_mapping.emplace_back(index_dims_mapping);
    indices_dist_attr_src.emplace_back(index_dist_attr_src);
    PADDLE_ENFORCE_EQ(
        index_ndim,
        index_dims_mapping.size(),
        phi::errors::InvalidArgument("The Tensor Index's rank [%d] and Index's "
                                     "dims_mapping size [%d] are not matched.",
                                     index_ndim,
                                     index_dims_mapping.size()));
  }

  auto value_shape = phi::vectorize(value.dims());
  int value_ndim = value_shape.size();
  TensorDistAttr value_dist_attr_src = value.dist_attr();
  std::vector<int64_t> value_dims_mapping = value_dist_attr_src.dims_mapping();
  PADDLE_ENFORCE_EQ(
      value_ndim,
      value_dims_mapping.size(),
      phi::errors::InvalidArgument("The Tensor Value's rank [%d] and Value's "
                                   "dims_mapping size [%d] are not matched.",
                                   value_ndim,
                                   value_dims_mapping.size()));

  // Step1: Build Einsum Notation
  // abc, {i, ..., i} , i -> abc
  std::string alphabet = "abcdefghjklmnopqrstuvwxyz";
  std::string x_axes = alphabet.substr(0, x_ndim);
  std::vector<std::string> indices_axes;
  for (int i = 0; i < x_ndim; ++i) {
    indices_axes.emplace_back("i");
  }
  std::string value_axes = "i";
  std::string out_axes = x_axes;

  // Step2: Sharding Propogation
  // step2.1 merge dims mappings of x, index, value.
  std::vector<std::pair<std::string, std::vector<int64_t>>>
      tensor_axes_to_dim_pairs;
  tensor_axes_to_dim_pairs.emplace_back(std::make_pair(x_axes, x_dims_mapping));
  for (int i = 0; i < x_ndim; ++i) {
    tensor_axes_to_dim_pairs.emplace_back(
        std::make_pair(indices_axes[i], indices_dims_mapping[i]));
  }
  tensor_axes_to_dim_pairs.emplace_back(
      std::make_pair(value_axes, value_dims_mapping));
  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors(tensor_axes_to_dim_pairs);
  // Step2.2 infer out dims mapping
  std::vector<int64_t> out_dims_mapping =
      GetDimsMappingForAxes(out_axes, axis_to_dim_map);
  // Step2.3 Update out dims mappings with merged one
  TensorDistAttr x_dist_attr_dst(x_dist_attr_src);
  std::vector<TensorDistAttr> indices_dist_attr_dst(indices_dist_attr_src);
  TensorDistAttr value_dist_attr_dst(value_dist_attr_src);
  TensorDistAttr out_dist_attr = CopyTensorDistAttrForOutput(x_dist_attr_src);
  x_dist_attr_dst.set_dims_mapping(
      GetDimsMappingForAxes(x_axes, axis_to_dim_map));
  for (int i = 0; i < x_ndim; ++i) {
    indices_dist_attr_dst[i].set_dims_mapping(
        GetDimsMappingForAxes(indices_axes[i], axis_to_dim_map));
  }
  value_dist_attr_dst.set_dims_mapping(
      GetDimsMappingForAxes(value_axes, axis_to_dim_map));
  out_dist_attr.set_dims_mapping(out_dims_mapping);

  VLOG(4) << "IndexPutSPMDRule InferForward:";
  VLOG(4) << "Input0 shape: [" << str_join(x_shape) << "] "
          << "src_dims_mapping: [" << str_join(x_dims_mapping) << "] "
          << "dst_dims_mapping: [" << str_join(x_dist_attr_dst.dims_mapping())
          << "]";
  for (int i = 0; i < x_ndim; ++i) {
    VLOG(4) << "Input1 shape: [" << str_join(indices_shape[i]) << "] "
            << "src_dims_mapping: [" << str_join(indices_dims_mapping[i])
            << "] " << "dst_dims_mapping: ["
            << str_join(indices_dist_attr_dst[i].dims_mapping()) << "]";
  }
  VLOG(4) << "Input2 shape: [" << str_join(value_shape) << "] "
          << "src_dims_mapping: [" << str_join(value_dims_mapping) << "] "
          << "dst_dims_mapping: ["
          << str_join(value_dist_attr_dst.dims_mapping()) << "]";
  VLOG(4) << "Output dims_mapping: [" + str_join(out_dims_mapping) + "]";

  return SpmdInfo({x_dist_attr_dst, indices_dist_attr_dst, value_dist_attr_dst},
                  {out_dist_attr});
}

SpmdInfo IndexPutInferSpmdReverse(const DistMetaTensor& x,
                                  const std::vector<DistMetaTensor>& indices,
                                  const DistMetaTensor& value,
                                  const DistMetaTensor& out,
                                  bool accumulate) {
  // Step0: Verify Input Args Based on IndexPut Logic
  auto x_shape = phi::vectorize(x.dims());
  int x_ndim = x_shape.size();
  TensorDistAttr x_dist_attr_src = x.dist_attr();

  std::vector<std::vector<int64_t>> indices_shape;
  std::vector<std::vector<int64_t>> indices_dims_mapping;
  std::vector<TensorDistAttr> indices_dist_attr_src;
  for (int i = 0; i < x_ndim; ++i) {
    auto index_shape = phi::vectorize(indices[i].dims());
    int index_ndim = index_shape.size();
    TensorDistAttr index_dist_attr_src = indices[i].dist_attr();
    std::vector<int64_t> index_dims_mapping =
        index_dist_attr_src.dims_mapping();
    indices_shape.emplace_back(index_shape);
    indices_dims_mapping.emplace_back(index_dims_mapping);
    indices_dist_attr_src.emplace_back(index_dist_attr_src);
    PADDLE_ENFORCE_EQ(
        index_ndim,
        index_dims_mapping.size(),
        phi::errors::InvalidArgument("The Tensor Index's rank [%d] and Index's "
                                     "dims_mapping size [%d] are not matched.",
                                     index_ndim,
                                     index_dims_mapping.size()));
  }

  auto value_shape = phi::vectorize(value.dims());
  int value_ndim = value_shape.size();
  TensorDistAttr value_dist_attr_src = value.dist_attr();
  std::vector<int64_t> value_dims_mapping = value_dist_attr_src.dims_mapping();
  PADDLE_ENFORCE_EQ(
      value_ndim,
      value_dims_mapping.size(),
      phi::errors::InvalidArgument("The Tensor Value's rank [%d] and Value's "
                                   "dims_mapping size [%d] are not matched.",
                                   value_ndim,
                                   value_dims_mapping.size()));

  auto out_shape = phi::vectorize(out.dims());
  int out_ndim = out_shape.size();
  TensorDistAttr out_dist_attr_src = out.dist_attr();
  std::vector<int64_t> out_dims_mapping = out_dist_attr_src.dims_mapping();
  PADDLE_ENFORCE_EQ(
      out_ndim,
      out_dims_mapping.size(),
      phi::errors::InvalidArgument("The Tensor Out's rank [%d] and Out's "
                                   "dims_mapping size [%d] are not matched.",
                                   out_ndim,
                                   out_dims_mapping.size()));

  // Step1: Build Einsum Notation
  // abc, {i, ..., i} , i -> abc
  std::string alphabet = "abcdefghjklmnopqrstuvwxyz";
  std::string x_axes = alphabet.substr(0, x_ndim);
  std::vector<std::string> indices_axes;
  for (int i = 0; i < x_ndim; ++i) {
    indices_axes.emplace_back("i");
  }
  std::string value_axes = "i";
  std::string out_axes = x_axes;

  // Step2: Sharding Propogation
  // step2.1 merge dims mappings of output, indices, value.
  std::vector<std::pair<std::string, std::vector<int64_t>>>
      tensor_axes_to_dim_pairs;
  tensor_axes_to_dim_pairs.emplace_back(
      std::make_pair(out_axes, out_dims_mapping));
  for (int i = 0; i < x_ndim; ++i) {
    tensor_axes_to_dim_pairs.emplace_back(
        std::make_pair(indices_axes[i], indices_dims_mapping[i]));
  }
  tensor_axes_to_dim_pairs.emplace_back(
      std::make_pair(value_axes, value_dims_mapping));
  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors(tensor_axes_to_dim_pairs);
  // Step2.2 infer x dims mapping
  std::vector<int64_t> x_dims_mapping =
      GetDimsMappingForAxes(x_axes, axis_to_dim_map);
  // Step2.3 Update x dims mappings with merged one
  TensorDistAttr x_dist_attr_dst(x_dist_attr_src);
  std::vector<TensorDistAttr> indices_dist_attr_dst(indices_dist_attr_src);
  TensorDistAttr value_dist_attr_dst(value_dist_attr_src);
  TensorDistAttr out_dist_attr_dst(out_dist_attr_src);
  x_dist_attr_dst.set_dims_mapping(x_dims_mapping);
  for (int i = 0; i < x_ndim; ++i) {
    indices_dist_attr_dst[i].set_dims_mapping(
        GetDimsMappingForAxes(indices_axes[i], axis_to_dim_map));
  }
  value_dist_attr_dst.set_dims_mapping(
      GetDimsMappingForAxes(value_axes, axis_to_dim_map));
  out_dist_attr_dst.set_dims_mapping(
      GetDimsMappingForAxes(out_axes, axis_to_dim_map));

  VLOG(4) << "IndexPutSPMDRule InferBackward:";
  VLOG(4) << "Input0 shape: [" << str_join(x_shape) << "] "
          << "src_dims_mapping: [" << str_join(x_dims_mapping) << "] "
          << "dst_dims_mapping: [" << str_join(x_dist_attr_dst.dims_mapping())
          << "]";
  for (int i = 0; i < x_ndim; ++i) {
    VLOG(4) << "Input1 shape: [" << str_join(indices_shape[i]) << "] "
            << "src_dims_mapping: [" << str_join(indices_dims_mapping[i])
            << "] " << "dst_dims_mapping: ["
            << str_join(indices_dist_attr_dst[i].dims_mapping()) << "]";
  }
  VLOG(4) << "Input2 shape: [" << str_join(value_shape) << "] "
          << "src_dims_mapping: [" << str_join(value_dims_mapping) << "] "
          << "dst_dims_mapping: ["
          << str_join(value_dist_attr_dst.dims_mapping()) << "]";
  VLOG(4) << "Output shape: [" << str_join(out_shape) << "] "
          << "src_dims_mapping: [" << str_join(out_dims_mapping) << "] "
          << "dst_dims_mapping: [" << str_join(out_dist_attr_dst.dims_mapping())
          << "]";

  return SpmdInfo({x_dist_attr_dst, indices_dist_attr_dst, value_dist_attr_dst},
                  {out_dist_attr_dst});
}

}  // namespace phi::distributed