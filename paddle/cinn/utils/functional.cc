// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/utils/functional.h"

#include "glog/logging.h"

namespace cinn {
namespace utils {

std::vector<int> GetPositiveAxes(const std::vector<int>& axes, int rank) {
  std::vector<int> new_axes(axes.size());
  for (int i = 0; i < axes.size(); ++i) {
    int axis = axes[i] + (axes[i] < 0 ? rank : 0);
    CHECK(axis >= 0 && (rank == 0 || axis < rank))
        << "The axis should in [" << -rank << ", " << rank << "), but axes["
        << i << "]=" << axes[i] << " not.";
    new_axes[i] = axis;
  }
  return new_axes;
}

int GetPositiveAxes(int axis, int rank) {
  int dim = axis + (axis < 0 ? rank : 0);
  CHECK(dim >= 0 && dim < rank)
      << "The axis should in [0, " << rank << "), but axis=" << axis << " not.";
  return dim;
}

}  // namespace utils
}  // namespace cinn
