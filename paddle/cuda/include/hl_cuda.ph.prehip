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


#ifndef HL_CUDA_PH_
#define HL_CUDA_PH_

#include <stdio.h>
#include <pthread.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cudnn.h>
#include "hl_base.h"

/**
 * @brief   hppl event.
 * @param   cuda event.
 */
struct _hl_event_st {
    cudaEvent_t     cu_event;       /* cuda event */
};

/**
 * @brief   global device resources.
 *
 * @param   *stream         device global stream.
 * @param   handle          devcie cublas handle.
 * @param   gen             device curand generator.
 * @param   cudnn_handle    cudnn handle.
 * @param   *gen_mutex      gen lock.
 */
typedef struct {
    cudaStream_t        *stream;
    cublasHandle_t      handle;
    curandGenerator_t   gen;
    cudnnHandle_t       cudnn_handle;
    pthread_mutex_t     *gen_mutex;
}_global_device_resources, *global_device_resources;

/*
 * @brief   thread device resources.
 *
 * @param   *stream         device thread stream.
 * @param   *gpu_mem        device memory.
 * @param   *cpu_mem        cpu memory.
 * @param    mem_event      device memory lock.
 */
typedef struct {
    cudaStream_t   *stream;
    real           *gpu_mem;
    real           *cpu_mem;
    cudaEvent_t    mem_event;
}_thread_device_resources, *thread_device_resources;

/*
 * @brief   hppl device properties.
 *
 * @param   device            device id.
 * @param   device_type       0.Nvidia, 1.AMD, 2.Intel.
 * @param   device_name[256]  device name.
 * @param   device_mem        total global memory.
 * @param   major             device compute capability.
 * @param   minor             device compute capability.
 * @param   is_local          local device or not.
 * @param   device_resources  device resources.
 */
typedef struct {
    int device;
    int device_type;
    char device_name[256];
    size_t device_mem;
    int major;
    int minor;
    bool is_local;
    global_device_resources device_resources;
} _hl_device_prop, *hl_device_prop;

/**
 * @brief   thread device resource allocation.
 *
 * create cuda stream and cuda event, allocate gpu
 * memory and host page-lock memory for threads.
 *
 * @param[in]   device      device number.
 * @param[out]  device_res  device properties.
 */
extern void hl_create_thread_resources(int device,
                                       thread_device_resources device_res);

/**
 * @brief   global device resource allocation.
 *
 * create cuda stream, initialize cublas, curand and cudnn.
 *
 * @param[out]   device_prop  device properties.
 */
extern void hl_create_global_resources(hl_device_prop device_prop);

#endif  /* HL_CUDA_PH_ */
