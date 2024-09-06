/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <string>
#include "cub/cub.cuh"

namespace phi {

static const float HALF_FLT_MAX = 65504.F;
static const float HALF_FLT_MIN = -65504.F;
static inline size_t AlignTo16(const size_t& input) {
  static constexpr int ALIGNMENT = 16;
  return ALIGNMENT * ((input + ALIGNMENT - 1) / ALIGNMENT);
}

class CubKeyValueSorter {
 public:
  CubKeyValueSorter() : num_experts_(0), num_bits_(sizeof(int) * 8) {}

  explicit CubKeyValueSorter(const int num_experts)
      : num_experts_(num_experts),
        num_bits_(static_cast<int>(log2(num_experts)) + 1) {}

  void update_num_experts(const int num_experts) {
    num_experts_ = num_experts;
    num_bits_ = static_cast<int>(log2(num_experts)) + 1;
  }

  size_t getWorkspaceSize(const size_t num_key_value_pairs,
                          bool descending = false) {
    num_key_value_pairs_ = num_key_value_pairs;
    size_t required_storage = 0;
    int* null_int = nullptr;
    if (descending) {
      cub::DeviceRadixSort::SortPairsDescending(NULL,
                                                required_storage,
                                                null_int,
                                                null_int,
                                                null_int,
                                                null_int,
                                                num_key_value_pairs,
                                                0,
                                                32);
    } else {
      cub::DeviceRadixSort::SortPairs(NULL,
                                      required_storage,
                                      null_int,
                                      null_int,
                                      null_int,
                                      null_int,
                                      num_key_value_pairs,
                                      0,
                                      num_bits_);
    }
    return required_storage;
  }

  template <typename KeyT>
  void run(void* workspace,
           const size_t workspace_size,
           const KeyT* keys_in,
           KeyT* keys_out,
           const int* values_in,
           int* values_out,
           const size_t num_key_value_pairs,
           bool descending,
           cudaStream_t stream) {
    size_t expected_ws_size = getWorkspaceSize(num_key_value_pairs);
    size_t actual_ws_size = workspace_size;

    if (expected_ws_size > workspace_size) {
      std::stringstream err_ss;
      err_ss << "[Error][CubKeyValueSorter::run]\n";
      err_ss << "Error. The allocated workspace is too small to run this "
                "problem.\n";
      err_ss << "Expected workspace size of at least " << expected_ws_size
             << " but got problem size " << workspace_size << "\n";
      throw std::runtime_error(err_ss.str());
    }
    if (descending) {
      cub::DeviceRadixSort::SortPairsDescending(workspace,
                                                actual_ws_size,
                                                keys_in,
                                                keys_out,
                                                values_in,
                                                values_out,
                                                num_key_value_pairs,
                                                0,
                                                32,
                                                stream);
    } else {
      cub::DeviceRadixSort::SortPairs(workspace,
                                      actual_ws_size,
                                      keys_in,
                                      keys_out,
                                      values_in,
                                      values_out,
                                      num_key_value_pairs,
                                      0,
                                      num_bits_,
                                      stream);
    }
  }

 private:
  size_t num_key_value_pairs_;
  int num_experts_;
  int num_bits_;
};

}  // namespace phi
