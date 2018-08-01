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
#include <sstream>  // std::stringstream
#include <string>

#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/var_type.h"
#include "paddle/fluid/framework/variable.h"

namespace paddle {
namespace framework {

template <typename T>
void GetArrayReadalbeData(int b, int e, int w, int summarize, void* data) {
  std::stringstream ss;
  auto* d = reinterpret_cast<T*>(data);

  int l_b = 0;
  int l_e = w < 2 * summarize ? w : summarize;
  int r_b = w - summarize;
  int r_e = w;

  for (int i = b; i < e; i++) {
    ss << "[";
    for (int m = l_b; m < l_e; m++) {
      int pos = i * w + m;
      ss << d[pos] if (pos < l_e - 1) { ss << ","; }
    }

    if (r_b > l_e) {
      ss << " ... ";
      for (int m = r_b; m < r_e; m++) {
        int pos = i * w + m;
        ss << d[pos];
        if (pos < r_e - 1) {
          ss << ",";
        }
      }
    }

    ss << "]";
    if (e - b > 1) {
      ss << std::endl;
    }
  }

  return ss.str();
}

template <typename T>
void GetMatrixReadableData(const std::string& message, const std::string& name,
                           int h, int w, int summarize, void* data) {
  std::stringstream ss;

  int t_b = 0;
  int t_e = h < 6 ? h : 3;
  int b_b = h - 3;
  int b_e = h;

  ss << message << " " << name << ":";
  ss << GetArrayReadalbeData<T>(t_b, t_e, w, summarize, data);

  if (b_b > t_e) {
    std::cout << "  ... " << std::endl;
    ss << GetArrayReadalbeData<T>(b_b, b_e, w, summarize, data);
  }

  return ss.str();
}

void GetVariableReadableData(framework::Variable* var, std::string name,
                             std::string message);

}  // namespace framework
}  // namespace paddle
