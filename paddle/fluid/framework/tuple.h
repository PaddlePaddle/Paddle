/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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

#include <stdexcept>
#include <string>
#include <vector>

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/var_desc.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {

typedef paddle::variant<int,
                        int64_t,
                        float,
                        double,
                        std::string,
                        phi::DenseTensor,
                        LoDTensor /*, ChannelHolder*/>
    ElementVar;

class Tuple {
 public:
  using ElementVars = std::vector<ElementVar>;

  Tuple(const std::vector<ElementVar>& var,
        const std::vector<VarDesc>& var_desc)
      : var_(var), var_desc_(var_desc) {}
  explicit Tuple(std::vector<ElementVar>& var) : var_(var) {}

  ElementVar get(int idx) const { return var_[idx]; }

  ElementVar& get(int idx) { return var_[idx]; }

  bool isSameType(const Tuple& t) const;

  size_t getSize() const { return var_.size(); }

 private:
  ElementVars var_;
  std::vector<VarDesc> var_desc_;
};

bool Tuple::isSameType(const Tuple& t) const {
  size_t tuple_size = getSize();
  if (tuple_size != t.getSize()) {
    return false;
  }
  for (size_t j = 0; j < tuple_size; ++j) {
    auto type1 = get(j).index();
    auto type2 = t.get(j).index();
    if (type1 != type2) return false;
  }
  return true;
}

Tuple* make_tuple(std::vector<ElementVar> tuple) { return new Tuple(tuple); }

}  // namespace framework
}  // namespace paddle
