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
#include "error.h"
#include "matrix.h"
#include "vector.h"

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
 * Arguments functions. Each argument means layer output. Arguments means a
 * array of arguemnt.
 */
typedef void* PD_Arguments;

/**
 * @brief PDArgsCreateNone Create a array of arguments, which size is zero.
 * @return Arguemnts
 */
PD_API PD_Arguments PDArgsCreateNone();

/**
 * @brief PDArgsDestroy Destroy the arguments
 * @param args arguments to destroy
 * @return paddle_error
 */
PD_API paddle_error PDArgsDestroy(PD_Arguments args);

/**
 * @brief PDArgsGetSize Get size of arguments array
 * @param [in] args arguments array
 * @param [out] size array size
 * @return paddle_error
 */
PD_API paddle_error PDArgsGetSize(PD_Arguments args, uint64_t* size);

/**
 * @brief PDArgsResize Resize a arguments array.
 * @param args arguments array.
 * @param size target size of array
 * @return paddle_error
 */
PD_API paddle_error PDArgsResize(PD_Arguments args, uint64_t size);

/**
 * @brief PDArgsSetValue Set value matrix of one argument in array, which index
 *        is `ID`.
 * @param args arguments array
 * @param ID array index
 * @param mat matrix pointer
 * @return paddle_error
 */
PD_API paddle_error PDArgsSetValue(PD_Arguments args,
                                   uint64_t ID,
                                   paddle_matrix mat);

/**
 * @brief PDArgsGetValue Get value matrix of one argument in array, which index
 *        is `ID`.
 * @param [in] args arguments array
 * @param [in] ID array index
 * @param [out] mat matrix pointer
 * @return paddle_error
 */
PD_API paddle_error PDArgsGetValue(PD_Arguments args,
                                   uint64_t ID,
                                   paddle_matrix mat);

/**
 * @brief PDArgsGetIds Get the integer vector of one argument in array, which
 *        index is `ID`.
 * @param args arguments array
 * @param ID array index
 * @param ids integer vector pointer
 * @return paddle_error
 */
PD_API paddle_error PDArgsGetIds(PD_Arguments args,
                                 uint64_t ID,
                                 paddle_ivector ids);

/**
 * @brief PDArgsSetIds Set the integer vector of one argument in array, which
 *        index is `ID`.
 * @param [in] args arguments array
 * @param [in] ID array index
 * @param [out] ids integer vector pointer
 * @return paddle_error
 */
PD_API paddle_error PDArgsSetIds(PD_Arguments args,
                                 uint64_t ID,
                                 paddle_ivector ids);

/**
 * @brief PDArgsSetSequenceStartPos Set sequence start position vector of one
 *        argument in array, which index is `ID`.
 * @param args arguments array
 * @param ID array index
 * @param seqPos sequence position array.
 * @return paddle_error
 */
PD_API paddle_error PDArgsSetSequenceStartPos(PD_Arguments args,
                                              uint64_t ID,
                                              paddle_ivector seqPos);
/**
 * @brief PDArgsGetSequenceStartPos Get sequence start position vector of one
 *        argument in array, which index is `ID`.
 * @param [in] args arguments array
 * @param [in] ID array index
 * @param [out] seqPos sequence position array
 * @return paddle_error
 */
PD_API paddle_error PDArgsGetSequenceStartPos(PD_Arguments args,
                                              uint64_t ID,
                                              paddle_ivector seqPos);

/**
 * @brief PDArgsSetSubSequenceStartPos Set sub-sequence start position vector of
 *        one argument in array, which index is `ID`.
 * @param args arguments array
 * @param ID array index
 * @param subSeqPos sub-sequence start position array.
 * @return paddle_error
 */
PD_API paddle_error PDArgsSetSubSequenceStartPos(PD_Arguments args,
                                                 uint64_t ID,
                                                 paddle_ivector subSeqPos);

/**
 * @brief PDArgsGetSubSequenceStartPos Get sub-sequence start position vector of
 *        one argument in array, which index is `ID`.
 * @param args arguments array
 * @param ID array index
 * @param subSeqPos sub-sequence start position array
 * @return paddle_error
 */
PD_API paddle_error PDArgsGetSubSequenceStartPos(PD_Arguments args,
                                                 uint64_t ID,
                                                 paddle_ivector subSeqPos);
/**
 * @brief GradientMachine means a neural network.
 */
typedef void* PD_GradientMachine;

/**
 * @brief PDGradientMachineCreateForPredict Create a gradient machine used for
 *        model inference.
 * @param [out] machine that used for model inference.
 * @param [in] modelConfigProtobuf
 * @param [in] size
 * @return paddle_error
 */
PD_API paddle_error PDGradientMachineCreateForPredict(
    PD_GradientMachine* machine, void* modelConfigProtobuf, int size);

/**
 * @brief PDGradientMachineLoadParameterFromDisk Load parameter from disk.
 * @param machine Gradient Machine.
 * @param path local directory path.
 * @return paddle_error
 */
PD_API paddle_error PDGradientMachineLoadParameterFromDisk(
    PD_GradientMachine machine, const char* path);

/**
 * @brief PDGradientMachineForward Forward a gradient machine
 * @param machine Gradient machine
 * @param inArgs input arguments
 * @param outArgs output arguments
 * @param isTrain is train or not
 * @return paddle_error
 */
PD_API paddle_error PDGradientMachineForward(PD_GradientMachine machine,
                                             PD_Arguments inArgs,
                                             PD_Arguments outArgs,
                                             bool isTrain);

/**
 * @brief PDGradientMachineCreateSharedParam Create a gradient machine, which
 *        parameters are shared from another gradient machine.
 * @param [in] origin gradient machine
 * @param [in] modelConfigProtobuf model config protobuf
 * @param [in] size of model config buffer.
 * @param [out] slave gradient machine, the output value.
 * @return paddle_error
 */
PD_API paddle_error
PDGradientMachineCreateSharedParam(PD_GradientMachine origin,
                                   void* modelConfigProtobuf,
                                   int size,
                                   PD_GradientMachine* slave);

/**
 * @brief PDGradientMachineDestroy Destroy a gradient machine
 * @param machine that need to destroy
 * @return paddle_error
 */
PD_API paddle_error PDGradientMachineDestroy(PD_GradientMachine machine);

/**
 * Initialize Paddle.
 */
PD_API paddle_error PDInit(int argc, char** argv);

#ifdef __cplusplus
}
#endif

#endif  // PADDLECAPI_H_
