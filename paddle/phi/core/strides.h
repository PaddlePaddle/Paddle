// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include <initializer_list>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "paddle/phi/core/ddim.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/utils/array.h"

namespace phi {

class Strides {
 public:
  Strides() {}

  static bool IsOpSupportStrides(const std::string& op_type);

  void InitStrides(const DDim& ddim);

  inline bool IsValiable() const { return is_valiable_; }

  inline bool IsContiguous() const { return is_contiguous_; }

  inline int64_t* GetMutable() { return strides_; }

  inline const int64_t* Get() const { return strides_; }

  inline void SetContiguous(bool contiguous) { is_contiguous_ = contiguous; }

  inline void SetRank(int rank) { rank_ = rank; }

 private:
  int64_t strides_[DDim::kMaxRank];
  int rank_{0};
  bool is_valiable_{false};
  bool is_contiguous_{true};
};

}  // namespace phi
