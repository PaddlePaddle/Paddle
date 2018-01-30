//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/framework/reader.h"

namespace paddle {
namespace framework {

DDim Reader::shape(int idx) const {
  PADDLE_ENFORCE_LT(
      idx, shapes_.size(),
      "Cannot get the %d'th shape, 'shapes_' only has %d elements.", idx,
      shapes_.size());
}

int RandomReader::ReadNext(std::vector<LoDTensor>* outs) {
  PADDLE_ENFORCE_EQ(
      shapes_.size(), outs.size(),
      "shapes_.size() is %d, while outs.size() is %d. They are not equal.",
      shapes_.size(), outs.size());
  std::minstd_rand engine;
  unsigned int seed = std::random_device()();
  engine.seed(seed);
  std::uniform_real_distribution<float> dist(min_, max_);
  for (int idx = 0; idx < shapes_.size(); ++idx) {
    DDim shape = shapes_[idx];
    LoDTensor* out = outs[idx];
    int64_t numel = out->numel();
    PADDLE_ENFORCE_EQ(product(shape), numel,
                      "The product of %d'th shape is %lld, while the "
                      "corresponding out's numel is %lld. They are not equal.",
                      idx, product(shape), numel);
    for (int64_t i = 0; i < numel, ++i) {
      out[i] = dist(engine);
    }
  }
  return 0;
}
}  // namespace framework
}  // namespace paddle
