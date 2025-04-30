/* Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include <numeric>
#include <vector>
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/empty_kernel.h"

namespace phi {
namespace sparse {

class ConvHostBuffer {
 public:
  ConvHostBuffer(const ConvHostBuffer&) = delete;
  ConvHostBuffer& operator=(const ConvHostBuffer&) = delete;

  static ConvHostBuffer& getInstance() {
    static ConvHostBuffer instance;
    return instance;
  }

  void set_host_buffer(int* buffer) { h_buffer_ = buffer; }

  int* get_host_buffer() {
    PADDLE_ENFORCE_EQ(offset_.empty(),
                      false,
                      ::common::errors::InvalidArgument(
                          "Sparse conv buffer offsets should not be empty."));
    return h_buffer_ + offset_[current_step_++];
  }

  void reset() { current_step_ = 0; }

  bool using_buffer() { return use_buffer_; }

  int get_buffer_size() { return buffer_size; }

  int get_max_bound() { return max_bound; }

  void init_from_config(const std::vector<std::vector<int>>& kernels,
                        const std::vector<std::vector<int>>& strides) {
    PADDLE_ENFORCE_EQ(kernels.size() == strides.size(),
                      true,
                      ::common::errors::InvalidArgument(
                          "The size of kernels should be equal to the "
                          "size of strides, but get kernel size:[%d], "
                          "strides size:[%d].",
                          kernels.size(),
                          strides.size()));
    buffer_size = 0;
    max_bound = 1;
    offset_.clear();
    for (size_t i = 0; i < kernels.size(); ++i) {
      int kernel_size = std::accumulate(
          kernels[i].begin(), kernels[i].end(), 1, std::multiplies<int>());
      buffer_size += 2 * kernel_size + 3;
      offset_.push_back(buffer_size - (2 * kernel_size + 3));
      int bound = 1;
      for (size_t j = 0; j < kernels[i].size(); ++j) {
        bound *= (kernels[i][j] + strides[i][j] - 1) / strides[i][j];
      }
      max_bound = std::max(max_bound, bound);
    }
    use_buffer_ = true;
  }

 private:
  ConvHostBuffer() {}
  ~ConvHostBuffer() {}

  int* h_buffer_;
  int buffer_size;
  std::vector<int> offset_;
  int current_step_{0};
  int max_bound;
  bool use_buffer_{false};
};

}  // namespace sparse
}  // namespace phi
