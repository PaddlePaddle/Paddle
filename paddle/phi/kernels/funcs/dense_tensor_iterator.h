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

#pragma once

#include <optional>

#include "paddle/common/ddim.h"
#include "paddle/phi/core/dense_tensor.h"

namespace phi {

struct DenseTensorIteratorConfig;
struct DenseTensorIterator;

enum struct FastSetupType : uint8_t { NONE, CONTIGUOUS };

/**
 * DenseOperandInfo: Used to store tensor-related information.
 * Contains metadata and details about tensors participating in operations.
 */
struct DenseOperandInfo {
  DenseOperandInfo() = default;
  inline explicit DenseOperandInfo(DenseTensor*&& t) {
    if (t->initialized()) {
      target_dtype = t->dtype();
      current_dtype = target_dtype;
    }
    tensor(std::move(t));
  }

  inline DenseOperandInfo(const DenseOperandInfo&) = default;
  inline DenseOperandInfo& operator=(const DenseOperandInfo&) = default;
  inline DenseOperandInfo(DenseOperandInfo&&) noexcept = default;
  inline DenseOperandInfo& operator=(DenseOperandInfo&&) noexcept = default;
  inline ~DenseOperandInfo() = default;

  void* data = nullptr;
  std::vector<int64_t> stride_bytes;
  DataType target_dtype = DataType::UNDEFINED;
  DataType current_dtype = DataType::UNDEFINED;
  bool is_output = false;
  bool will_resize = false;
  bool is_read_write = false;
  bool is_const = false;
  bool is_type_defined() const { return target_dtype != DataType::UNDEFINED; }
  DenseTensor& tensor() const { return *tensor_base_; }
  void tensor(DenseTensor*&& tensor);

 private:
  DenseTensor* tensor_base_;
};

/**
 * DenseTensorIteratorBase: Base class for DenseTensorIterator.
 * Defines and supports the key functions used by DenseTensorIterator.
 */
struct DenseTensorIteratorBase {
  void build(DenseTensorIteratorConfig&);
  int ndim() const { return static_cast<int>(shape_.size()); }
  const std::vector<int64_t>& shape() const { return shape_; }
  int64_t numel() const;
  int ntensors() const { return static_cast<int>(operands_.size()); }
  bool is_contiguous() const;
  const std::vector<int64_t>& strides(int64_t arg) const {
    return operands_[arg].stride_bytes;
  }
  const void* data_ptr(int64_t arg) const;

 protected:
  void populate_operands(DenseTensorIteratorConfig&);
  void compute_shape(const DenseTensorIteratorConfig&);
  void compute_strides(const DenseTensorIteratorConfig&);
  void reorder_dimensions();
  void permute_dimensions(std::vector<int64_t> perm);
  void allocate_or_resize_outputs();
  bool fast_set_up(const DenseTensorIteratorConfig&);
  FastSetupType compute_fast_setup_type(const DenseTensorIteratorConfig&);
  void coalesce_dimensions();

 protected:
  std::vector<int64_t> shape_;
  std::vector<int64_t> perm_;
  bool has_coalesced_dimensions_ = false;
  int num_outputs_ = 0;
  bool all_ops_same_shape_ = false;
  bool all_ops_are_scalars_ = false;

 public:
  std::vector<DenseOperandInfo> operands_;
  std::vector<int64_t> compatible_stride(int64_t element_size) const;
  std::vector<int64_t> invert_perm(std::vector<int64_t> input) const;
  virtual void set_output_raw_strided(int64_t output_idx,
                                      std::vector<int64_t> sizes,
                                      std::vector<int64_t> strides);
  bool is_reduction_ = false;
};

/**
 * DenseTensorIterator: Used for preprocessing metadata of tensors participating
 * in computation. Can be directly used as OffsetCalculator input parameter to
 * assist with index calculations.
 */
struct DenseTensorIterator final : public DenseTensorIteratorBase {
  DenseTensorIterator() : DenseTensorIteratorBase() {}
  DenseTensorIterator(const DenseTensorIteratorBase& iter)
      : DenseTensorIteratorBase(iter) {}

  void set_output_raw_strided(int64_t output_idx,
                              std::vector<int64_t> sizes,
                              std::vector<int64_t> strides) override;
};

/**
 * DenseTensorIteratorConfig: Used to configure tensors and computation rules
 * for DenseTensorIterator
 *
 * This class configures the tensors participating in computation and the
 * operation rules for DenseTensorIterator. Usage example:
 *
 * DenseTensorIteratorConfig config;
 * // Add tensors participating in computation
 * // Set whether to use specific methods in TensorIterator
 * config.add_output(a);
 * config.add_const_input(b);
 * config.add_const_input(c);
 *
 * // Calculate the common broadcast shape and transformed strides for each
 * dimension DenseTensorIterator iter = config.build();
 */
struct DenseTensorIteratorConfig final {
 public:
  friend struct DenseTensorIteratorBase;
  friend struct DenseTensorIterator;

  DenseTensorIteratorConfig() = default;
  DenseTensorIteratorConfig(DenseTensorIteratorConfig&&) = default;
  DenseTensorIteratorConfig& operator=(DenseTensorIteratorConfig&&) = default;
  ~DenseTensorIteratorConfig() = default;

  DenseTensorIteratorConfig& add_output(const DenseTensor& output) {
    return add_borrowed_output(output);
  }
  DenseTensorIteratorConfig& add_input(const DenseTensor& input) {
    return add_borrowed_input(input);
  }
  DenseTensorIteratorConfig& add_const_input(const DenseTensor& input) {
    return add_borrowed_const_input(input);
  }

  DenseTensorIteratorConfig& add_output(DenseTensor&& output) = delete;
  DenseTensorIteratorConfig& add_input(DenseTensor&& input) = delete;
  DenseTensorIteratorConfig& add_const_input(DenseTensor&& input) = delete;

  DenseTensorIteratorConfig& add_borrowed_output(const DenseTensor& output);
  DenseTensorIteratorConfig& add_borrowed_input(const DenseTensor& input);
  DenseTensorIteratorConfig& add_borrowed_const_input(const DenseTensor& input);

  DenseTensorIteratorConfig& add_borrowed_output(DenseTensor&& output) = delete;
  DenseTensorIteratorConfig& add_borrowed_input(DenseTensor&& input) = delete;
  DenseTensorIteratorConfig& add_borrowed_const_input(DenseTensor&& input) =
      delete;

  DenseTensorIteratorConfig& resize_outputs(bool resize_outputs) {
    resize_outputs_ = resize_outputs;
    return *this;
  }

  DenseTensorIterator build() {
    DenseTensorIterator iter;
    iter.build(*this);
    return iter;
  }

 private:
  std::vector<const DenseTensor*> tensors_;
  std::vector<size_t> const_tensor_indices_;
  int num_outputs_ = 0;
  int num_inputs_ = 0;

  std::optional<std::vector<int64_t>> static_shape_ = std::nullopt;
  bool is_reduction_ = false;
  bool resize_outputs_ = true;
};

}  // namespace phi
