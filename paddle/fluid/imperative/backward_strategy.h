// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

//
// Created by Jiabin on 2019-04-25.
//
#pragma once
#ifndef PADDLE_BACKWARDSTRATEGY_H
#define PADDLE_BACKWARDSTRATEGY_H

#endif  // PADDLE_BACKWARDSTRATEGY_H

namespace paddle {
namespace imperative {
namespace detail {

class BackwardStrategy {
 public:
  /* DyGraph now support two kinds of backward strategy, one is sorted sum
   * gradient, another is sum gradient once they are created */
  // TODO(jiabin): add more Strategy when we support
  bool sorted_sum_gradient_{false};
};

}  // namespace detail
}  // namespace imperative
}  // namespace paddle
