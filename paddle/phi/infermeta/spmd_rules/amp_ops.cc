// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/infermeta/spmd_rules/amp_ops.h"

#include <vector>
#include "glog/logging.h"

#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi {
namespace distributed {
SpmdInfo CheckFiniteAndUnscaleSpmd(const std::vector<DistMetaTensor>& xs,
                                   const DistMetaTensor& scale) {
  std::vector<TensorDistAttr> xs_attrs;
  bool splited = false;
  for (auto& x : xs) {
    auto dist_attr = x.dist_attr();
    dist_attr.clean_partial_status();
    xs_attrs.emplace_back(dist_attr);
    if (splited) {
      continue;
    }
    auto dims_mapping = dist_attr.dims_mapping();
    for (auto& m : dims_mapping) {
      if (m != -1) {
        splited = true;
      }
    }
  }
  TensorDistAttr found_infinite_attr =
      CopyTensorDistAttrForOutput(scale.dist_attr());
  if (splited) {
    found_infinite_attr.set_partial(true);
  }
  return {{xs_attrs, scale.dist_attr()}, {xs_attrs, found_infinite_attr}};
}
}  // namespace distributed
}  // namespace phi
