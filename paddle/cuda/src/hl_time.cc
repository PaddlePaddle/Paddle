/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "hl_time.h"
#include <stdlib.h>
#include <chrono>
#include <cstdint>
#include <iostream>

using std::chrono::high_resolution_clock;

int64_t getCurrentTimeStick() {
  high_resolution_clock::time_point tp = high_resolution_clock::now();
  high_resolution_clock::duration dtn = tp.time_since_epoch();
  return dtn.count();
}
