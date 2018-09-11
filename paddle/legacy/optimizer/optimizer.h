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

#pragma once

#include <stdbool.h>
#include <stdint.h>

/**
 * @brief optimizer library in independent with other module
 * which will be used in :
 * Case A, the gradient optimized locally on the trainer.
 *
 * Case B, the gradient optimized on the parameter server.
 */

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  PADDLE_ELEMENT_TYPE_INT32 = 0,
  PADDLE_ELEMENT_TYPE_UINT32 = 1,
  PADDLE_ELEMENT_TYPE_INT64 = 2,
  PADDLE_ELEMENT_TYPE_UINT64 = 3,
  PADDLE_ELEMENT_TYPE_FLOAT32 = 4,
  PADDLE_ELEMENT_TYPE_FLOAT64 = 5,
} paddle_element_type;

/**
 * @brief execution status code
 */
const int32_t PADDLE_SUCCESS = 0;
const int32_t PADDLE_ERROR = -1;

typedef struct paddle_optimizer paddle_optimizer;
/**
 * this group interface called in order :
 * 1. create optimizer with config
 * 2. set weights
 * 3. update_parameter
 * 4. get_weights
 * 5. release optimizer
 */

/**
 *  @brief create optimizer with proto_config
 *  @param config_proto, optimizer protobuf, see OptimizerConfig.proto in detail
 *  @return return optimizer instance
 */
paddle_optimizer* paddle_create_optimizer(const unsigned char* config_proto,
                                          const int config_proto_len,
                                          const paddle_element_type data_type,
                                          void* param_buffer,
                                          int num_bytes,
                                          const char* state,
                                          const int state_len);

/**
 *  @brief release optimizer
 *  @param optimizer
 *  @return return exec status
 */
int paddle_release_optimizer(paddle_optimizer* o);

/**
 *  @brief optimizer instance
 *  @param datatype of gradient and parameter
 *  @param gradient, calculate by optimzizer caller.
 *       TODO(zhihong): just pass loss to reduce communicate overhead.
 *                     Project Adam Ms'14 paper for detail
 *  @param num_bytes, gradient size
 *  @return return exec status
 */
int paddle_update_parameter(paddle_optimizer* o,
                            const paddle_element_type data_type,
                            const void* gradient,
                            int num_bytes);

/**
 *  @brief optimizer for get parameter buffer
 *  @param param_buffer, initilized parameter buffer
 *  @return return content length
 */
int paddle_optimizer_get_weights(paddle_optimizer* o, void** param_buffer);

/**
 *  @brief optimzizer for saving training state
 *  @param training state for receive SerializeState
 *  @return return state_buffer length
 */
int paddle_optimizer_get_state(paddle_optimizer* o, const char** state);

#ifdef __cplusplus
}
#endif
