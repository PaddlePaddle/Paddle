// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#ifdef PADDLE_WITH_NCCL

namespace paddle {
namespace framework {
class Variable;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace imperative {

struct ParallelStrategy;

void AllReduce(const framework::Variable &src, framework::Variable *dst,
               const ParallelStrategy &strategy);

void AllReduce(const framework::Variable &src, framework::Variable *dst,
               const ParallelStrategy &strategy, int ring_id,
               bool use_calc_stream);

}  // namespace imperative
}  // namespace paddle

#endif
