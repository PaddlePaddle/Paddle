// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/funcs/dense_tensor_iterator.h"

namespace phi {

void DenseOperandInfo::tensor(DenseTensor*&& tensor) {
  tensor_base_ = std::move(tensor);
}

DenseTensorIteratorConfig& DenseTensorIteratorConfig::add_borrowed_output(
    const DenseTensor& output) {
  PADDLE_ENFORCE_EQ(num_inputs_,
                    0,
                    "Keep in mind that you have to add all outputs first "
                    "before adding any input.");
  tensors_.push_back(&output);
  num_outputs_++;
  return *this;
}

DenseTensorIteratorConfig& DenseTensorIteratorConfig::add_borrowed_input(
    const DenseTensor& input) {
  tensors_.push_back(&input);
  num_inputs_++;
  return *this;
}

DenseTensorIteratorConfig& DenseTensorIteratorConfig::add_borrowed_const_input(
    const DenseTensor& input) {
  const_tensor_indices_.push_back(tensors_.size());
  tensors_.push_back(&input);
  num_inputs_++;
  return *this;
}

void DenseTensorIteratorBase::reorder_dimensions() {
  perm_.resize(ndim());
  if (ndim() == 1) {
    perm_[0] = 0;
    return;
  }
  std::iota(perm_.rbegin(), perm_.rend(), 0);
  auto should_swap = [&](size_t dim0, size_t dim1) {
    for (auto arg = 0; arg < ntensors(); arg++) {
      if (operands_[arg].stride_bytes.empty() || operands_[arg].will_resize) {
        continue;
      }
      int64_t stride0 = operands_[arg].stride_bytes[dim0];
      int64_t stride1 = operands_[arg].stride_bytes[dim1];
      if (is_reduction_ && operands_[arg].is_output) {
        if ((stride0 == 0) != (stride1 == 0)) {
          return stride1 == 0 ? 1 : -1;
        }
      }
      if (stride0 == 0 || stride1 == 0) {
        continue;
      } else if (stride0 < stride1) {
        return -1;
      } else if (stride0 > stride1) {
        return 1;
      } else {
        auto t_dim0 = shape_[dim0];
        auto t_dim1 = shape_[dim1];
        if (t_dim0 > t_dim1) {
          return 1;
        }
      }
    }
    return 0;
  };
  for (auto i = 1; i < ndim(); i++) {
    int dim1 = i;
    for (int dim0 = i - 1; dim0 >= 0; dim0--) {
      int comparison = should_swap(perm_[dim0], perm_[dim1]);
      if (comparison > 0) {
        std::swap(perm_[dim0], perm_[dim1]);
        dim1 = dim0;
      } else if (comparison < 0) {
        break;
      }
    }
  }
  permute_dimensions(perm_);
}

void DenseTensorIteratorBase::permute_dimensions(std::vector<int64_t> perm) {
  PADDLE_ENFORCE_EQ(
      perm.size(),
      static_cast<unsigned>(ndim()),
      "perm.size() must equal to ndim in DenseDenseTensorIterator");
  auto reorder = [perm](std::vector<int64_t> data) {
    auto res = std::vector<int64_t>(data.size(), 0);
    for (size_t i = 0; i < perm.size(); i++) {
      res[i] = data[perm[i]];
    }
    return res;
  };
  shape_ = reorder(shape_);
  for (auto& op : operands_) {
    if (!op.stride_bytes.empty()) {
      op.stride_bytes = reorder(op.stride_bytes);
    }
  }
}

std::vector<int64_t> DenseTensorIteratorBase::compatible_stride(
    int64_t element_size) const {
  std::vector<int64_t> stride;
  int64_t next_stride = element_size;
  for (auto dim = 0; dim < ndim(); dim++) {
    stride.push_back(next_stride);
    next_stride *= shape_[dim];
  }
  return stride;
}

std::vector<int64_t> DenseTensorIteratorBase::invert_perm(
    std::vector<int64_t> input) const {
  auto res = std::vector<int64_t>(input.size());
  for (auto dim = 0; dim < ndim(); dim++) {
    res[perm_[dim]] = input[dim];
  }
  return res;
}

void DenseTensorIteratorBase::allocate_or_resize_outputs() {
  for (auto i = 0; i < num_outputs_; i++) {
    auto& op = operands_[i];
    bool valid_stride = op.tensor().strides().size() == -1 ? false : true;
    bool reduce_pass = false;
    if (is_reduction_ && !valid_stride && op.is_output) {
      reduce_pass = true;
    }
    if (!reduce_pass &&
        (!op.tensor().initialized() || op.will_resize || !valid_stride)) {
      auto element_size = phi::SizeOf(op.tensor().dtype());
      op.stride_bytes = compatible_stride(static_cast<int64_t>(element_size));
      bool inverted = true;
      for (auto j = 0; j < ndim(); j++) {
        if (perm_[j] != ndim() - j - 1) {
          inverted = false;
          break;
        }
      }
      auto tensor_shape = invert_perm(shape_);
      if (inverted) {
        set_output_raw_strided(i, tensor_shape, {});
      } else {
        auto tensor_stride = invert_perm(op.stride_bytes);
        for (auto dim = 0; dim < ndim(); dim++) {
          tensor_stride[dim] /= static_cast<int64_t>(element_size);
        }
        set_output_raw_strided(i, tensor_shape, tensor_stride);
      }
      op.current_dtype = op.target_dtype;
    } else if (op.tensor().initialized()) {
      set_output_raw_strided(
          i, common::vectorize<int64_t>(op.tensor().dims()), {});
    }
  }
}

void DenseTensorIteratorBase::set_output_raw_strided(
    int64_t output_idx,
    std::vector<int64_t> sizes,
    std::vector<int64_t> strides) {
  PADDLE_THROW(
      common::errors::Fatal("Virtual Set Output Stride, Unsupported!"));
}

void DenseTensorIterator::set_output_raw_strided(int64_t output_idx,
                                                 std::vector<int64_t> sizes,
                                                 std::vector<int64_t> strides) {
  auto& op = operands_[output_idx];
  bool valid_stride = op.tensor().strides().size() == -1 ? false : true;
  if (!op.tensor().initialized() || !valid_stride) {
    if (strides.empty()) {
      auto meta = op.tensor().meta();
      auto new_dims = common::make_ddim(sizes);
      auto new_strides = meta.calc_strides(new_dims);
      meta.dims = new_dims;
      meta.strides = new_strides;
      op.tensor().set_meta(meta);
    } else {
      auto meta = op.tensor().meta();
      auto new_dims = common::make_ddim(sizes);
      auto new_strides = common::make_ddim(strides);
      meta.dims = new_dims;
      meta.strides = new_strides;
      op.tensor().set_meta(meta);
    }
    op.current_dtype = op.target_dtype;
  } else if (op.will_resize) {
    PADDLE_THROW(common::errors::Fatal("Opreator Reize not Implemented!"));
  }
}

void DenseTensorIteratorBase::coalesce_dimensions() {
  if (ndim() <= 1) {
    return;
  }
  auto can_coalesce = [&](int dim0, int dim1) {
    auto shape0 = shape_[dim0];
    auto shape1 = shape_[dim1];
    if (shape0 == 1 || shape1 == 1) {
      return true;
    }
    for (auto i = 0; i < ntensors(); i++) {
      auto& stride = operands_[i].stride_bytes;
      if (shape0 * stride[dim0] != stride[dim1]) {
        return false;
      }
    }
    return true;
  };
  auto replace_stride = [&](int dim0, int dim1) {
    for (auto i = 0; i < ntensors(); i++) {
      auto& stride = operands_[i].stride_bytes;
      stride[dim0] = stride[dim1];
    }
  };
  int prev_dim = 0;
  for (auto dim = 1; dim < ndim(); dim++) {
    if (can_coalesce(prev_dim, dim)) {
      if (shape_[prev_dim] == 1) {
        replace_stride(prev_dim, dim);
      }
      shape_[prev_dim] *= shape_[dim];
    } else {
      prev_dim++;
      if (prev_dim != dim) {
        replace_stride(prev_dim, dim);
        shape_[prev_dim] = shape_[dim];
      }
    }
  }
  shape_.resize(prev_dim + 1);
  for (auto i = 0; i < ntensors(); i++) {
    operands_[i].stride_bytes.resize(ndim());
  }
  has_coalesced_dimensions_ = true;
}

int64_t DenseTensorIteratorBase::numel() const {
  int64_t numel = 1;
  for (int64_t size : shape_) {
    numel *= size;
  }
  return numel;
}

const void* DenseTensorIteratorBase::data_ptr(int64_t arg) const {
  return static_cast<void*>(operands_[arg].tensor().data());
}

static inline std::vector<int64_t> infer_size_dimvector(
    std::vector<int64_t> a, std::vector<int64_t> b) {
  auto dimsA = a.size();
  auto dimsB = b.size();
  auto ndim = dimsA > dimsB ? dimsA : dimsB;
  std::vector<int64_t> expandedSizes = std::vector<int64_t>(ndim, 0);
  for (int64_t i = ndim - 1; i >= 0; --i) {
    int64_t offset = ndim - 1 - i;
    int64_t dimA = dimsA - 1 - offset;
    int64_t dimB = dimsB - 1 - offset;
    auto sizeA = (dimA >= 0) ? a[dimA] : 1;
    auto sizeB = (dimB >= 0) ? b[dimB] : 1;
    expandedSizes[i] = sizeA == 1 ? sizeB : sizeA;
  }
  return expandedSizes;
}

void DenseTensorIteratorBase::populate_operands(
    DenseTensorIteratorConfig& config) {
  for (size_t idx = 0; idx < config.tensors_.size(); idx++) {
    auto& tensor = config.tensors_[idx];
    operands_.emplace_back(std::move(const_cast<DenseTensor*>(tensor)));
    if (idx < config.num_outputs_) {
      operands_[idx].is_output = true;
    }
  }
  num_outputs_ = config.num_outputs_;
}

FastSetupType DenseTensorIteratorBase::compute_fast_setup_type(
    const DenseTensorIteratorConfig& config) {
  if (is_reduction_ || !all_ops_same_shape_) {
    return FastSetupType::NONE;
  }
  bool is_contiguous = true;
  for (const auto& op : operands_) {
    if (op.tensor().initialized() && !op.will_resize) {
      is_contiguous &= op.tensor().meta().is_contiguous();
    }
  }
  if (is_contiguous) {
    return FastSetupType::CONTIGUOUS;
  }
  return FastSetupType::NONE;
}

bool DenseTensorIteratorBase::fast_set_up(
    const DenseTensorIteratorConfig& config) {
  FastSetupType setup_type = compute_fast_setup_type(config);
  if (setup_type == FastSetupType::NONE) {
    return false;
  }
  switch (setup_type) {
    case FastSetupType::CONTIGUOUS: {
      for (auto i = 0; i < num_outputs_; i++) {
        set_output_raw_strided(i, shape_, {});
      }
      break;
    }
    default:
      PADDLE_THROW(common::errors::Fatal("Unsupported Fast Setup Type!"));
  }
  if (ndim() > 1) {
    has_coalesced_dimensions_ = true;
  }
  if (ndim() >= 1) {
    shape_[0] = numel();
    shape_.resize(1);
  }
  for (auto& op : operands_) {
    auto element_size_in_bytes = phi::SizeOf(op.tensor().dtype());
    op.stride_bytes.resize(ndim());
    if (ndim() > 0) {
      op.stride_bytes[0] = element_size_in_bytes;
    }
  }
  return true;
}

int DenseTensorIteratorBase::num_reduce_dims() const {
  int count = 0;
  for (int dim = 0; dim < ndim(); dim++) {
    if (operands_[0].stride_bytes[dim] == 0) {
      count++;
    }
  }
  return count;
}

int64_t DenseTensorIteratorBase::num_output_elements() const {
  int64_t elem = 1;
  for (int dim = 0; dim < ndim(); dim++) {
    if (operands_[0].stride_bytes[dim] != 0 || shape_[dim] == 0) {
      elem *= shape_[dim];
    }
  }
  return elem;
}

void DenseTensorIteratorBase::compute_shape(
    const DenseTensorIteratorConfig& config) {
  all_ops_same_shape_ = true;
  bool has_scalars = false;
  bool has_tensors = false;
  for (auto& op : operands_) {
    bool valid_stride = op.tensor().strides().size() == -1 ? false : true;
    if (!op.tensor().initialized() || !valid_stride) continue;
    if (config.resize_outputs_ && op.is_output) continue;
    auto shape = common::vectorize<int64_t>(op.tensor().dims());
    if (shape.empty()) {
      has_scalars = true;
    } else {
      has_tensors = true;
    }
    if (has_scalars && has_tensors) {
      all_ops_same_shape_ = false;
    }
    if (shape_.empty()) {
      shape_ = shape;
    } else if (!(shape == shape_)) {
      all_ops_same_shape_ = false;
      shape_ = infer_size_dimvector(shape_, shape);
    }
  }
  all_ops_are_scalars_ = !has_tensors;
}

void DenseTensorIteratorBase::compute_strides(
    const DenseTensorIteratorConfig& config) {
  for (auto& op : operands_) {
    bool valid_stride = op.tensor().strides().size() == -1 ? false : true;

    bool reduce_pass = false;

    std::vector<int64_t> tmp_shape =
        common::vectorize<int64_t>(op.tensor().dims());
    std::vector<int64_t> tmp_stride =
        common::vectorize<int64_t>(op.tensor().strides());

    if (is_reduction_ && !valid_stride && op.is_output) {
      tmp_stride = std::vector<int64_t>(shape_.size(), 0);
      tmp_shape = std::vector<int64_t>(shape_.size(), 1);
      reduce_pass = true;
    }

    if (reduce_pass ||
        op.tensor().initialized() && !op.will_resize && valid_stride) {
      std::vector<int64_t> original_shape;
      original_shape = config.static_shape_
                           ? shape_
                           : common::vectorize<int64_t>(op.tensor().dims());
      if (op.is_output && reduce_pass) original_shape = tmp_shape;
      std::vector<int64_t> original_stride;
      original_stride = common::vectorize<int64_t>(op.tensor().strides());
      if (op.is_output && reduce_pass) original_stride = tmp_stride;
      auto element_size_in_bytes = phi::SizeOf(op.tensor().dtype());
      auto offset = ndim() - original_shape.size();
      if (offset > 0)
        op.stride_bytes.resize(ndim(), 0);
      else
        op.stride_bytes.resize(ndim());
      for (size_t i = 0; i < original_shape.size(); i++) {
        if (original_shape[i] == 1 && shape_[offset + i] != 1) {
          op.stride_bytes[offset + i] = 0;
        } else {
          op.stride_bytes[offset + i] =
              original_stride[i] * element_size_in_bytes;
        }
      }
    }
  }
}

void DenseTensorIteratorBase::build(DenseTensorIteratorConfig& config) {
  is_reduction_ = config.is_reduction_;
  populate_operands(config);
  compute_shape(config);
  if (!fast_set_up(config)) {
    compute_strides(config);
    reorder_dimensions();
    allocate_or_resize_outputs();
    coalesce_dimensions();
  }
}
}  // namespace phi
