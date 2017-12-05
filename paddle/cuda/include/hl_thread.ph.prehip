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

#ifndef HL_THREAD_PH_
#define HL_THREAD_PH_

#include <stdio.h>
#include <pthread.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cudnn.h>
#include "hl_base.h"

/**
 * @brief   Thread resource structure.
 *
 * @param   stream[HPPL_STREAM_END] Stream for thread.
 * @param   handle                  Cublas Handle.
 * @param   gen                     Curand Generator.
 * @param   cudnn_handle            Cudnn handle.
 * @param   cudnn_desc              Cudnn image descriptor.
 * @param   *gen_mutex              Gen lock.
 * @param   *gpu_mem                HPPL GPU Memory.
 * @param   *cpu_mem                HPPL CPU Memory.
 * @param   event                   gpu_mem event.
 * @param   device                  Thread device context.
 * @param   major                   Compute capability.
 * @param   is_init                 Thread init or not.
 */
typedef struct {
    cudaStream_t             stream[HPPL_STREAM_END];
    cublasHandle_t           handle;
    curandGenerator_t        gen;
    cudnnHandle_t            cudnn_handle;
    cudnnTensorDescriptor_t  cudnn_desc;
    pthread_mutex_t          *gen_mutex;
    real                     *gpu_mem;
    real                     *cpu_mem;
    cudaEvent_t              event;
    int                      device;
    int                      major;
    bool                     is_init;
} _hl_thread_resource, *hl_thread_resource;

extern __thread _hl_thread_resource t_resource;

/**
 * @brief   Initialize cudnn.
 *
 * @param   cudnn_handle  Cudnn handle.
 * @param   stream        Cudnn stream.
 */
extern void hl_cudnn_init(cudnnHandle_t *cudnn_handle, cudaStream_t stream);

/**
 * @brief   Initialize cublas.
 *
 * @param   cublas_handle  Cublas handle.
 * @param   stream         Cuda stream.
 */
extern void hl_cublas_init(cublasHandle_t *cublas_handle, cudaStream_t stream);

/**
 * @brief   Initialize cudnn tensor descriptor.
 *
 * @param   cudnn_desc    Cudnn tensor descriptor.
 */

extern void hl_cudnn_desc_init(cudnnTensorDescriptor_t*  cudnn_desc);

#endif  /* HL_THREAD_PH_ */
