// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include <thread>

#include "paddle/cinn/runtime/cinn_runtime.h"

extern "C" {

int max_concurrency();

/**
 * @brief The callback function to execute a parallel lambda
 * @param task_id the task id of the function.
 * @param num_task The Number of tasks to launch. If 0, it means to launch
 *           with all available threads.
 * @param datas The closure datas.
 */
typedef int (*FCINNParallelLambda)(int task_id, int num_task, void* datas);

/**
 * @brief Backend function for running parallel jobs.
 *
 * @param flambda The parallel function to be launched.
 * @param datas The closure datas.
 * @param num_task The Number of tasks to launch. If 0, it means to launch
 *           with all available threads.
 *
 * @return 0 when no error is thrown, -1 when failure happens
 */
int cinn_backend_parallel_launch(FCINNParallelLambda flambda,
                                 void* datas,
                                 int num_task);

}  // extern "C"
