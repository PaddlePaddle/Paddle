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
#include "paddle/utils/small_vector.h"

namespace phi {

struct DenseTensorIteratorConfig;
struct DenseTensorIterator;
struct Tensor32BitSplitter;

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
  int64_t num_output_elements() const;
  int noutputs() const { return num_outputs_; }
  int num_reduce_dims() const;
  const std::vector<int64_t>& strides(int64_t arg) const {
    return operands_[arg].stride_bytes;
  }
  DataType dtype(int64_t arg = 0) const { return operands_[arg].current_dtype; }
  std::vector<int64_t> view_offsets() const { return view_offsets_; }
  void* data_ptr(int64_t arg) const;
  bool should_accumulate() const { return accumulate_; }
  bool is_final_output() const { return final_output_; }
  int get_dim_to_split() const;
  bool is_dim_reduced(int dim) const;
  std::unique_ptr<DenseTensorIterator> split(int dim);

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
  void narrow(int dim, int64_t start, int64_t size);

 protected:
  std::vector<int64_t> shape_;
  std::vector<int64_t> perm_;
  std::vector<int64_t> view_offsets_;
  bool has_coalesced_dimensions_ = false;
  size_t num_outputs_ = 0;
  bool all_ops_same_shape_ = false;
  bool all_ops_are_scalars_ = false;

 public:
  std::vector<DenseOperandInfo> operands_;
  std::vector<int64_t> compatible_stride(int64_t element_size) const;
  std::vector<int64_t> invert_perm(std::vector<int64_t> input) const;
  bool can_use_32bit_indexing() const;
  Tensor32BitSplitter with_32bit_indexing() const;
  virtual void set_output_raw_strided(int64_t output_idx,
                                      std::vector<int64_t> sizes,
                                      std::vector<int64_t> strides);
  bool is_reduction_ = false;
  bool is_alloc_out_ = false;
  bool accumulate_ = false;
  bool final_output_ = true;
};

/**
 * DenseTensorIterator: Used for preprocessing metadata of tensors
 * participating in computation. Can be directly used as OffsetCalculator
 * input parameter to assist with index calculations.
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

  DenseTensorIteratorConfig& is_reduction(const bool _is_reduction) {
    is_reduction_ = _is_reduction;
    return *this;
  }

  DenseTensorIterator build() {
    DenseTensorIterator iter;
    iter.build(*this);
    return iter;
  }

  bool is_alloc_out_ = false;

 private:
  std::vector<const DenseTensor*> tensors_;
  std::vector<size_t> const_tensor_indices_;
  size_t num_outputs_ = 0;
  size_t num_inputs_ = 0;

  std::optional<std::vector<int64_t>> static_shape_ = std::nullopt;
  bool is_reduction_ = false;
  bool resize_outputs_ = false;
};

struct DimIter {
  DimIter(std::vector<int64_t> shape, int64_t start, int64_t end);

  void iter_to_next(const std::array<int64_t, 2>& step);
  bool iter_to_end() const;
  std::array<int64_t, 2> iter_for_step() const;

  std::vector<int64_t> shape;
  int64_t start;
  int64_t end;
  paddle::small_vector<int64_t, 4> values;
  int64_t offset;
};

struct Tensor32BitSplitter {
  struct iterator {
    iterator() = default;
    explicit iterator(const DenseTensorIteratorBase& iter);
    iterator(iterator&&) = default;
    iterator& operator=(iterator&&) = default;
    ~iterator() = default;

    DenseTensorIterator& operator*() const;
    iterator& operator++();

    bool operator==(const iterator& other) const {
      return this == &other ||
             (iterator_stack_.empty() && other.iterator_stack_.empty());
    }

    bool operator!=(const iterator& other) const { return !(*this == other); }

    std::vector<std::unique_ptr<DenseTensorIterator>> iterator_stack_;
  };

  explicit Tensor32BitSplitter(const DenseTensorIteratorBase& iter)
      : source_iterator_(iter) {}

  iterator begin() const;
  iterator end() const;

 private:
  const DenseTensorIteratorBase& source_iterator_;
};

}  // namespace phi
