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
#pragma once

#include <cassert>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "paddle_api.h"  // NOLINT

namespace paddle {
namespace contrib {
// Configurations for Anakin engine.
struct AnakinConfig : public PaddlePredictor::Config {
  enum TargetType { NVGPU = 0, X86, MLU, BM };
  int device_id{0};
  std::string model_file;
  std::map<std::string, std::vector<int>> init_inputs_shape;
  int init_batch_size{-1};
  bool re_allocable{true};
  int max_stream{4};
  int data_stream_id{0};
  int compute_stream_id{0};
  char* model_buf_p{nullptr};
  size_t model_buf_len{0};
  TargetType target_type;
#ifdef ANAKIN_MLU_PLACE
  int model_parallel{8};
  int data_parallel{1};
  bool op_fuse{false};
  bool sparse{false};
#endif
};

}  // namespace contrib
}  // namespace paddle
