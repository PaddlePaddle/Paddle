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

#include "paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/infer_sym_slice_utils.h"

namespace paddle::dialect::slice_utils {

void SliceDfsImpl(const ExprVec &datas,
                  const std::vector<int64_t> &shape,
                  int64_t axis,
                  int64_t start,
                  int64_t end,
                  int64_t cur_visit_axis,
                  int offset,
                  ExprVec *result) {
  int64_t begin = 0;
  int64_t stop = shape.at(cur_visit_axis);
  if (cur_visit_axis == axis) {
    begin = start;
    stop = end;
  }
  const int64_t cur_stride = std::accumulate(shape.begin() + cur_visit_axis + 1,
                                             shape.end(),
                                             1,
                                             std::multiplies<int64_t>());
  for (int64_t i = begin; i < stop; ++i) {
    const int64_t cur_offset = offset + i * cur_stride;
    // last dim
    if (cur_visit_axis == static_cast<int64_t>(shape.size() - 1)) {
      result->push_back(datas[cur_offset]);
    } else {
      SliceDfsImpl(datas,
                   shape,
                   axis,
                   start,
                   end,
                   cur_visit_axis + 1,
                   cur_offset,
                   result);
    }
  }
}

ExprVec SimpleSlice(const ExprVec &datas,
                    const std::vector<int64_t> &shape,
                    int64_t axis,
                    int64_t start,
                    int64_t end) {
  ExprVec result;
  SliceDfsImpl(datas, shape, axis, start, end, 0, 0, &result);
  return result;
}

}  // namespace paddle::dialect::slice_utils
