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

/**
 * @brief Create an none int vector. It just a handler and store nothing. Used
 *        to get output from other api.
 * @return None int vector.
 */
PD_API PD_IVector PDIVecCreateNone();

/**
 * @brief PDIVectorCreate create a paddle int vector
 * @param array: input array.
 * @param size: input array size.
 * @param copy: memory copy or just use same memory. True if copy.
 * @param useGPU: True if use GPU
 * @return PD_Error
 */
PD_API PD_IVector PDIVectorCreate(int* array,
                                  uint64_t size,
                                  bool copy,
                                  bool useGPU);

/**
 * @brief PDIVecDestroy destory an int vector.
 * @param ivec vector to be destoried.
 * @return PD_Error
 */
PD_API PD_Error PDIVecDestroy(PD_IVector ivec);

/**
 * @brief PDIVectorGet get raw buffer stored inside this int vector. It could be
 *        GPU memory if this int vector is stored in GPU.
 * @param [in] ivec int vector
 * @param [out] buffer the return buffer pointer.
 * @return PD_Error
 */
PD_API PD_Error PDIVectorGet(PD_IVector ivec, int** buffer);

/**
 * @brief PDIVectorResize resize the int vector.
 * @param [in] ivec: int vector
 * @param [in] size: size to change
 * @return PD_Error
 */
PD_API PD_Error PDIVectorResize(PD_IVector ivec, uint64_t size);

/**
 * @brief PDIVectorGetSize get the size of int vector.
 * @param [in] ivec: int vector
 * @param [out] size: return size of this int vector.
 * @return PD_Error
 */
PD_API PD_Error PDIVectorGetSize(PD_IVector ivec, uint64_t* size);

/**
 * Matrix functions. Return will be a PD_Error type.
 */
typedef void* PD_Matrix;

/**
 * @brief PDMatCreate Create a dense matrix
 * @param height matrix height.
 * @param width matrix width
 * @param useGpu use GPU of not
 * @return Matrix handler
 */
PD_API PD_Matrix PDMatCreate(uint64_t height, uint64_t width, bool useGpu);

/**
 * @brief PDMatDestroy Destroy a matrix.
 * @param mat
 * @return PD_Error
 */
PD_API PD_Error PDMatDestroy(PD_Matrix mat);

/**
 * @brief PDMatCopyToRow Copy a row to matrix.
 * @param mat Target Matrix
 * @param rowID Index of row
 * @param rowArray Row data.
 * @return PD_Error
 */
PD_API PD_Error PDMatCopyToRow(PD_Matrix mat,
                               uint64_t rowID,
                               pd_real* rowArray);

/**
 * @brief PDMatGetRow Get raw row buffer from matrix
 * @param [in] mat Target matrix
 * @param [in] rowID Index of row.
 * @param [out] rawRowBuffer Row Buffer
 * @return PD_Error
 */
PD_API PD_Error PDMatGetRow(PD_Matrix mat,
                            uint64_t rowID,
                            pd_real** rawRowBuffer);

/**
 * @brief PDMatCreateNone Create None Matrix
 * @return
 */
PD_API PD_Matrix PDMatCreateNone();

/**
 * @brief PDMatGetShape get the shape of matrix
 * @param mat target matrix
 * @param height The height of matrix
 * @param width The width of matrix
 * @return PD_Error
 */
PD_API PD_Error PDMatGetShape(PD_Matrix mat, uint64_t* height, uint64_t* width);

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
 * @return PD_Error
 */
PD_API PD_Error PDArgsDestroy(PD_Arguments args);

/**
 * @brief PDArgsGetSize Get size of arguments array
 * @param [in] args arguments array
 * @param [out] size array size
 * @return PD_Error
 */
PD_API PD_Error PDArgsGetSize(PD_Arguments args, uint64_t* size);

/**
 * @brief PDArgsResize Resize a arguments array.
 * @param args arguments array.
 * @param size target size of array
 * @return PD_Error
 */
PD_API PD_Error PDArgsResize(PD_Arguments args, uint64_t size);

/**
 * @brief PDArgsSetValue Set value matrix of one argument in array, which index
 *        is `ID`.
 * @param args arguments array
 * @param ID array index
 * @param mat matrix pointer
 * @return PD_Error
 */
PD_API PD_Error PDArgsSetValue(PD_Arguments args, uint64_t ID, PD_Matrix mat);

/**
 * @brief PDArgsGetValue Get value matrix of one argument in array, which index
 *        is `ID`.
 * @param [in] args arguments array
 * @param [in] ID array index
 * @param [out] mat matrix pointer
 * @return PD_Error
 */
PD_API PD_Error PDArgsGetValue(PD_Arguments args, uint64_t ID, PD_Matrix mat);

/**
 * @brief PDArgsGetIds Get the integer vector of one argument in array, which
 *        index is `ID`.
 * @param args arguments array
 * @param ID array index
 * @param ids integer vector pointer
 * @return PD_Error
 */
PD_API PD_Error PDArgsGetIds(PD_Arguments args, uint64_t ID, PD_IVector ids);

/**
 * @brief PDArgsSetIds Set the integer vector of one argument in array, which
 *        index is `ID`.
 * @param [in] args arguments array
 * @param [in] ID array index
 * @param [out] ids integer vector pointer
 * @return PD_Error
 */
PD_API PD_Error PDArgsSetIds(PD_Arguments args, uint64_t ID, PD_IVector ids);

/**
 * @brief PDArgsSetSequenceStartPos Set sequence start position vector of one
 *        argument in array, which index is `ID`.
 * @param args arguments array
 * @param ID array index
 * @param seqPos sequence position array.
 * @return PD_Error
 */
PD_API PD_Error PDArgsSetSequenceStartPos(PD_Arguments args,
                                          uint64_t ID,
                                          PD_IVector seqPos);
/**
 * @brief PDArgsGetSequenceStartPos Get sequence start position vector of one
 *        argument in array, which index is `ID`.
 * @param [in] args arguments array
 * @param [in] ID array index
 * @param [out] seqPos sequence position array
 * @return PD_Error
 */
PD_API PD_Error PDArgsGetSequenceStartPos(PD_Arguments args,
                                          uint64_t ID,
                                          PD_IVector seqPos);

/**
 * @brief PDArgsSetSubSequenceStartPos Set sub-sequence start position vector of
 *        one argument in array, which index is `ID`.
 * @param args arguments array
 * @param ID array index
 * @param subSeqPos sub-sequence start position array.
 * @return PD_Error
 */
PD_API PD_Error PDArgsSetSubSequenceStartPos(PD_Arguments args,
                                             uint64_t ID,
                                             PD_IVector subSeqPos);

/**
 * @brief PDArgsGetSubSequenceStartPos Get sub-sequence start position vector of
 *        one argument in array, which index is `ID`.
 * @param args arguments array
 * @param ID array index
 * @param subSeqPos sub-sequence start position array
 * @return PD_Error
 */
PD_API PD_Error PDArgsGetSubSequenceStartPos(PD_Arguments args,
                                             uint64_t ID,
                                             PD_IVector subSeqPos);
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
 * @return PD_Error
 */
PD_API PD_Error PDGradientMachineCreateForPredict(PD_GradientMachine* machine,
                                                  void* modelConfigProtobuf,
                                                  int size);

/**
 * @brief PDGradientMachineLoadParameterFromDisk Load parameter from disk.
 * @param machine Gradient Machine.
 * @param path local directory path.
 * @return PD_Error
 */
PD_API PD_Error PDGradientMachineLoadParameterFromDisk(
    PD_GradientMachine machine, const char* path);

/**
 * @brief PDGradientMachineForward Forward a gradient machine
 * @param machine Gradient machine
 * @param inArgs input arguments
 * @param outArgs output arguments
 * @param isTrain is train or not
 * @return PD_Error
 */
PD_API PD_Error PDGradientMachineForward(PD_GradientMachine machine,
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
 * @return PD_Error
 */
PD_API PD_Error PDGradientMachineCreateSharedParam(PD_GradientMachine origin,
                                                   void* modelConfigProtobuf,
                                                   int size,
                                                   PD_GradientMachine* slave);

/**
 * @brief PDGradientMachineDestroy Destroy a gradient machine
 * @param machine that need to destroy
 * @return PD_Error
 */
PD_API PD_Error PDGradientMachineDestroy(PD_GradientMachine machine);

/**
 * Initialize Paddle.
 */
PD_API PD_Error PDInit(int argc, char** argv);

#ifdef __cplusplus
}
#endif

#endif  // PADDLECAPI_H_
