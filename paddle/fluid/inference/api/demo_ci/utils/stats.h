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

#include <gflags/gflags.h>
#include <vector>
#include "paddle/fluid/inference/paddle_inference_api.h"

namespace paddle {

struct Stats {
  explicit Stats(int32_t batch_size, int32_t skip_batch_num)
      : batch_size(batch_size), skip_batch_num(skip_batch_num){};

  void Gather(const std::vector<PaddleTensor>& output_slots,
              double batch_time,
              int iter);
  void Postprocess(double total_time_sec, int total_samples);

  std::vector<double> latencies;
  std::vector<float> infer_accs1;
  std::vector<float> infer_accs5;
  std::vector<double> fpses;
  int32_t batch_size;
  int32_t skip_batch_num;
};

}  // namespace paddle
