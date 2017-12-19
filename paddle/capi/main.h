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

#ifndef __PADDLE_CAPI_MAIN_H__
#define __PADDLE_CAPI_MAIN_H__
#include "config.h"
#include "error.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize Paddle.
 */
PD_API paddle_error paddle_init(int argc, char** argv);

/**
 * set the thread count of OpenMP or Eigen when run on android or ios device.
 * @note thread count will be set to the smaller between n and cpu cores.
 */
PD_API paddle_error paddle_set_num_threads(int n);

/*
 * get the thread count of OpenMP or Eigen
 */
PD_API paddle_error paddle_get_num_threads(int* n);

/**
 * Initialize the thread environment of Paddle.
 * @note it is requisite for GPU runs but optional for CPU runs.
 *       For GPU runs, all threads will run on the same GPU devices.
 */
PD_API paddle_error paddle_init_thread();

#ifdef __cplusplus
}
#endif

#endif
