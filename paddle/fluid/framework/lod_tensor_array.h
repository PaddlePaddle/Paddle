/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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
#include <vector>
#include "paddle/fluid/framework/lod_tensor.h"

namespace paddle {
namespace framework {
//#ifndef PADDLE_ON_INFERENCE

// using LoDTensorArray = std::vector<LoDTensor>;

//#else  // PADDLE_ON_INFERENCE

/*
 * For inference, for the variables will not be cleaned after each batch
 * finished, the original LoDTensorArray acts weried, and might keep `push_back`
 * items and result accumulate memory leak in deployment. So we make a new
 * implementation for easier control and specific optimization for inference.
 */
class LoDTensorArray
    : public std::vector<LoDTensor, std::allocator<LoDTensor>> {
 public:
  using value_type = LoDTensor;
  using base_type = std::vector<LoDTensor, std::allocator<LoDTensor>>;

  void emplace_back(const value_type &v) {
    base_type::emplace_back(v);
    if (size() > 30UL) {
      //LOG(WARNING) << "There is a LoDTensorArray get's a length " << size();
    }
  }

  void emplace_back(value_type &&v) {
    base_type::emplace_back(std::move(v));
    if (size() > 30UL) {
      //LOG(WARNING) << "There is a LoDTensorArray get's a length " << size();
    }
  }

  void push_back(const value_type& v) {
    base_type::push_back(v);
    if (size() > 30UL) {
      //LOG(WARNING) << "There is a LoDTensorArray get's a length " << size();
    }
  }

  void push_back(value_type&& v) {
    base_type::push_back(std::move(v));
    if (size() > 30UL) {
      //LOG(WARNING) << "There is a LoDTensorArray get's a length " << size();
    }
  }

  LoDTensorArray() : std::vector<value_type>() {}
  LoDTensorArray(size_t count,
                 const value_type &value = value_type())  // NOLINT
      : std::vector<value_type>(count, value) {}
  LoDTensorArray(std::initializer_list<value_type> init)
      : std::vector<value_type>(init) {}
  LoDTensorArray(const std::vector<value_type> &other)
      : std::vector<value_type>(other) {}  // NOLINT
  LoDTensorArray(const LoDTensorArray &other)
      : std::vector<value_type>(other) {}
  LoDTensorArray(LoDTensorArray &&other)
      : std::vector<value_type>(std::move(other)) {}
  LoDTensorArray &operator=(const LoDTensorArray &other) {
    this->assign(other.begin(), other.end());
    return *this;
  }
  LoDTensorArray &operator=(const std::vector<value_type> &other) {
    this->assign(other.begin(), other.end());
    return *this;
  }
};

//#endif  // PADDLE_ON_INFERENCE

}  // namespace framework
}  // namespace paddle
