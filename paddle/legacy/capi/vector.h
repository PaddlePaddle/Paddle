/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifndef __PADDLE_CAPI_VECTOR_H__
#define __PADDLE_CAPI_VECTOR_H__

#include <stdbool.h>
#include <stdint.h>
#include "config.h"
#include "error.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Int Vector Functions. Return will be a paddle_error type.
 */
typedef void* paddle_ivector;

/**
 * @brief Create an none int vector. It just a handler and store nothing. Used
 *        to get output from other api.
 * @return None int vector.
 */
PD_API paddle_ivector paddle_ivector_create_none();

/**
 * @brief paddle_ivector_create create a paddle int vector
 * @param array: input array.
 * @param size: input array size.
 * @param copy: memory copy or just use same memory. True if copy.
 * @param useGPU: True if use GPU
 * @return paddle_error
 */
PD_API paddle_ivector paddle_ivector_create(int* array,
                                            uint64_t size,
                                            bool copy,
                                            bool useGPU);

/**
 * @brief paddle_ivector_destroy destory an int vector.
 * @param ivec vector to be destoried.
 * @return paddle_error
 */
PD_API paddle_error paddle_ivector_destroy(paddle_ivector ivec);

/**
 * @brief paddle_ivector_get get raw buffer stored inside this int vector. It
 * could be GPU memory if this int vector is stored in GPU.
 * @param [in] ivec int vector
 * @param [out] buffer the return buffer pointer.
 * @return paddle_error
 */
PD_API paddle_error paddle_ivector_get(paddle_ivector ivec, int** buffer);

/**
 * @brief paddle_ivector_resize resize the int vector.
 * @param [in] ivec: int vector
 * @param [in] size: size to change
 * @return paddle_error
 */
PD_API paddle_error paddle_ivector_resize(paddle_ivector ivec, uint64_t size);

/**
 * @brief paddle_ivector_get_size get the size of int vector.
 * @param [in] ivec: int vector
 * @param [out] size: return size of this int vector.
 * @return paddle_error
 */
PD_API paddle_error paddle_ivector_get_size(paddle_ivector ivec,
                                            uint64_t* size);

#ifdef __cplusplus
}
#endif

#endif
