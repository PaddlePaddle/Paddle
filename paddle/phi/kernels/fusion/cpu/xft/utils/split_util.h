// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <utility>
#include "./compile_util.h"

class SplitUtil {
 public:
  static std::pair<int, int> getTaskRange(int N, int splits, int splitIdx) {
    int startId, endId;

    if (N % splits == 0) {
      int tasksPerSplit = N / splits;
      startId = splitIdx * tasksPerSplit;
      endId = startId + tasksPerSplit;
    } else {
      int baseTasksPerSplit = N / splits;
      int remainingTasks = N % splits;

      // Each split has (baseTasksPerSplit + 1) tasks
      if (splitIdx < remainingTasks) {
        int tasksPerSplit = baseTasksPerSplit + 1;
        startId = splitIdx * tasksPerSplit;
        endId = startId + tasksPerSplit;
      } else {
        // Each split has 'baseTasksPerSplit' tasks
        int taskOffset = (baseTasksPerSplit + 1) * remainingTasks;
        startId = taskOffset + (splitIdx - remainingTasks) * baseTasksPerSplit;
        endId = startId + baseTasksPerSplit;
      }
    }

    return std::make_pair(startId, endId);
  }

  // Split the task with a minimum granularity of 'gran'
  // For example, if N=64, gran=16, splits=3, then it will be split into 32 + 16
  // + 16
  static std::pair<int, int> getTaskRange(int N,
                                          int gran,
                                          int splits,
                                          int splitIdx) {
    REQUIRES(N % gran == 0,
             "N (%d) need to be multiple of granularity (%d).",
             N,
             gran);

    auto ret = getTaskRange(N / gran, splits, splitIdx);

    return std::make_pair(ret.first * gran, ret.second * gran);
  }
};
