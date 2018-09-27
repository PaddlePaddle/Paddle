// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>

namespace paddle {
namespace framework {
namespace details {

struct BuildStrategy {
  // ParallelExecutor supports two modes of ReduceStrategy, kAllReduce and
  // kReduce, for CPU and GPU. If you use kAllReduce, different threads
  // optimize their parameters separately. If you use kReduce, the optimizations
  // of parameters are distributed to different threads.
  // For example, a model has 100 parameters and is running with four threads,
  // if you choose kAllReduce, every thread is to optimize 100 parameters
  // separately, if you choose kReduce, every thread is to optimize 25
  // parameters.
  // Of particular note is, if you use kReduce when using CPU training,
  // all the parameters are shared between different threads. This feature will
  // save memory.
  // FIXME(zcd): The result of the two modes(kAllReduce and kReduce) maybe not
  // equal for GPU. Because, the result of the different order of summing maybe
  // different, for example, the result of `a+b+c+d` may be different with the
  // result of `c+a+b+d`.
  // For GPU, the implementation of kAllReduce and kReduce is adopted NCCL,
  // so the result of kAllReduce and kReduce maybe not equal.
  // For CPU, if you want to fix the order of summing to make the result
  // of kAllReduce and kReduce no diff, you can add
  // `FLAGS_cpu_deterministic=true` to env.
  enum class ReduceStrategy { kAllReduce = 0, kReduce = 1 };

  enum class GradientScaleStrategy {
    kCoeffNumDevice = 0,
    kOne = 1,
    kCustomized = 2,
  };

  ReduceStrategy reduce_{ReduceStrategy::kAllReduce};
  GradientScaleStrategy gradient_scale_{GradientScaleStrategy::kCoeffNumDevice};

  std::string debug_graphviz_path_{""};

  bool fuse_elewise_add_act_ops_{false};

  bool enable_data_balance_{false};
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
