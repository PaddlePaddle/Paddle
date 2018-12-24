/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include <sys/time.h>
#include "paddle/fluid/framework/scope.h"

namespace paddle {
namespace platform {

// A Standard Timer implementation for debugging
class Timer {
 public:
  Timer() { Reset(); }
  void Reset();
  void Start();
  void Pause();
  void Resume();
  int Count();
  double ElapsedUS();
  double ElapsedMS();
  double ElapsedSec();

 private:
  struct timeval _start;
  struct timeval _now;
  int _count;
  int _elapsed;
  bool _paused;

  int64_t Tickus();
};

}  // namespace platform
}  // namespace paddle
