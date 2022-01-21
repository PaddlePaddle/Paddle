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
#include <stdlib.h>

#include "paddle/pten/backends/dynload/port.h"

#ifdef _WIN32
static unsigned sleep(unsigned seconds) {
  Sleep(seconds * 1000);
  return 0;
}
#endif

namespace paddle {
namespace platform {

// A Standard Timer implementation for debugging
class Timer {
 public:
  // a timer class for profiling
  // Reset() will be called during initialization
  // all timing variables will be set 0 in Reset()
  Timer() { Reset(); }
  void Reset();
  void Start();
  void Pause();
  // Resume will get current system time
  void Resume();
  int Count();
  // return elapsed time in us
  double ElapsedUS();
  // return elapsed time in ms
  double ElapsedMS();
  // return elapsed time in sec
  double ElapsedSec();

 private:
  struct timeval _start;
  struct timeval _now;
  int _count;
  int64_t _elapsed;
  bool _paused;

  // get us difference between start and now
  int64_t Tickus();
};

}  // namespace platform
}  // namespace paddle
