#pragma once

#include <utility>
#include "compile_util.h"

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
            }

            // Each split has 'baseTasksPerSplit' tasks
            else {
                int taskOffset = (baseTasksPerSplit + 1) * remainingTasks;
                startId = taskOffset + (splitIdx - remainingTasks) * baseTasksPerSplit;
                endId = startId + baseTasksPerSplit;
            }
        }

        return std::make_pair(startId, endId);
    }

    // Split the task with a minimum granularity of 'gran'
    // For example, if N=64, gran=16, splits=3, then it will be split into 32 + 16 + 16
    static std::pair<int, int> getTaskRange(int N, int gran, int splits, int splitIdx) {
        REQUIRES(N % gran == 0, "N (%d) need to be multiple of granularity (%d).", N, gran);

        auto ret = getTaskRange(N / gran, splits, splitIdx);

        return std::make_pair(ret.first * gran, ret.second * gran);
    }
};