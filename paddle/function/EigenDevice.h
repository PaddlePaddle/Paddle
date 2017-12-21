/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License. */

#pragma once

#include "unsupported/Eigen/CXX11/Tensor"

namespace paddle {

int GetCpuCount();

const Eigen::ThreadPoolDevice& GetThreadPoolDevice();

class ThreadsNumManager {
public:
  static void Set(int n) { manage_threads_num(SetAction, &n); }

  static int Get() {
    int n = 1;
    manage_threads_num(GetAction, &n);
    return n;
  }

private:
  enum Action { GetAction, SetAction };

  static void manage_threads_num(Action a, int* n) {
    static int num_threads = GetCpuCount();
    if (a == SetAction) {
      if (*n > 0 && *n < num_threads) {
        num_threads = *n;
      }
    } else {
      *n = num_threads;
    }
  }
};

}  // namespace paddle
