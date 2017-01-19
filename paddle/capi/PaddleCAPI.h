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

#ifndef PADDLECAPI_H_
#define PADDLECAPI_H_
#include <stdbool.h>
#include <stdint.h>
#include "config.h"

// Since we only support linux and macos in compile, always use clang or
// gcc 4.8+. DLL_IMPORT/DLL_EXPORT is as simple as below.
#define PD_API __attribute__((visibility("default")))

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Paddle C API. It will replace SWIG as Multiple Language API for model
 * training & inference. Currently it is only used in model infernece.
 *
 * NOTE: This is an experimental API, it could be changed.
 */

/**
 * Error Type for Paddle API.
 */
typedef enum {
  kPD_NO_ERROR = 0,
  kPD_NULLPTR = 1,
  kPD_OUT_OF_RANGE = 2,
  kPD_PROTOBUF_ERROR = 3,
  kPD_UNDEFINED_ERROR = -1,
} PD_Error;

/**
 * Int Vector Functions. Return will be a PD_Error type.
 */
typedef void* PD_IVector;

PD_API int PDIVecCreateNone(PD_IVector* ivec);

PD_API int PDIVecDestroy(PD_IVector ivec);

PD_API int PDIVectorGet(PD_IVector ivec, int** buffer);

/**
 * Matrix functions. Return will be a PD_Error type.
 */
typedef void* PD_Matrix;

PD_API int PDMatCreate(PD_Matrix* mat,
                       uint64_t height,
                       uint64_t width,
                       bool useGpu);

PD_API int PDMatDestroy(PD_Matrix mat);

PD_API int PDMatCopyToRow(PD_Matrix mat, uint64_t rowID, pd_real* rowArray);

PD_API int PDMatGetRow(PD_Matrix mat, uint64_t rowID, pd_real** rawRowBuffer);

PD_API int PDMatCreateNone(PD_Matrix* mat);

PD_API int PDMatGetShape(PD_Matrix mat, uint64_t* height, uint64_t* width);

/**
 * Arguments functions. Each argument means layer output. Arguments means a
 * array of arguemnt.
 */
typedef void* PD_Arguments;

PD_API int PDArgsCreateNone(PD_Arguments* args);

PD_API int PDArgsDestroy(PD_Arguments args);

PD_API int PDArgsGetSize(PD_Arguments args, uint64_t* size);

PD_API int PDArgsResize(PD_Arguments args, uint64_t size);

PD_API int PDArgsSetValue(PD_Arguments args, uint64_t ID, PD_Matrix mat);

PD_API int PDArgsGetValue(PD_Arguments args, uint64_t ID, PD_Matrix mat);

PD_API int PDArgsGetIds(PD_Arguments args, uint64_t ID, PD_IVector ids);

/**
 * @brief GradientMachine means a neural network.
 */
typedef void* PD_GradientMachine;

PD_API int PDGradientMachineCreateForPredict(PD_GradientMachine* machine,
                                             void* modelConfigProtobuf,
                                             int size);

PD_API int PDGradientMachineLoadParameterFromDisk(PD_GradientMachine machine,
                                                  const char* path);

PD_API int PDGradientMachineForward(PD_GradientMachine machine,
                                    PD_Arguments inArgs,
                                    PD_Arguments outArgs,
                                    bool isTrain);

PD_API int PDGradientMachineCreateSharedParam(PD_GradientMachine origin,
                                              void* modelConfigProtobuf,
                                              int size,
                                              PD_GradientMachine* slave);

PD_API int PDGradientMachineDestroy(PD_GradientMachine machine);

/**
 * Initialize Paddle.
 */
PD_API int PDInit(int argc, char** argv);

#ifdef __cplusplus
}
#endif

#endif  // PADDLECAPI_H_
