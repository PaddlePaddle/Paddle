#ifndef __PADDLE_PADDLE_CAPI_PADDLECAPI_H_INCLUDED__
#define __PADDLE_PADDLE_CAPI_PADDLECAPI_H_INCLUDED__
#include <stdbool.h>
#include <stdint.h>
#include "config.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  kPD_NO_ERROR = 0,
  kPD_NULLPTR = 1,
  kPD_OUT_OF_RANGE = 2,
  kPD_PROTOBUF_ERROR = 3,
  kPD_UNDEFINED_ERROR = -1,
} PD_Error;

typedef void* PD_IVector;

int PDIVecCreateNone(PD_IVector* ivec);

int PDIVecDestroy(PD_IVector ivec);

int PDIVectorGet(PD_IVector ivec, int** buffer);

typedef void* PD_Matrix;

int PDMatCreate(PD_Matrix* mat, uint64_t height, uint64_t width, bool useGpu);

int PDMatDestroy(PD_Matrix mat);

int PDMatCopyToRow(PD_Matrix mat, uint64_t rowID, pd_real* rowArray);

int PDMatGetRow(PD_Matrix mat, uint64_t rowID, pd_real** rawRowBuffer);

int PDMatCreateNone(PD_Matrix* mat);

int PDMatGetShape(PD_Matrix mat, uint64_t* height, uint64_t* width);

typedef void* PD_Arguments;

int PDArgsCreateNone(PD_Arguments* args);

int PDArgsDestroy(PD_Arguments args);

int PDArgsGetSize(PD_Arguments args, uint64_t* size);

int PDArgsResize(PD_Arguments args, uint64_t size);

int PDArgsSetValue(PD_Arguments args, uint64_t ID, PD_Matrix mat);

int PDArgsGetValue(PD_Arguments args, uint64_t ID, PD_Matrix mat);

int PDArgsGetIds(PD_Arguments args, uint64_t ID, PD_IVector ids);

typedef void* PD_GradiemtMachine;

int PDGradientMachineCreateForPredict(PD_GradiemtMachine* machine,
                                      void* modelConfigProtobuf,
                                      int size);

int PDGradientMachineLoadParameterFromDisk(PD_GradiemtMachine machine,
                                           const char* path);

int PDGradientMachineForward(PD_GradiemtMachine machine,
                             PD_Arguments inArgs,
                             PD_Arguments outArgs,
                             bool isTrain);

int PDGradientMachineCreateSharedParam(PD_GradiemtMachine origin,
                                       void* modelConfigProtobuf,
                                       int size,
                                       PD_GradiemtMachine* slave);

int PDGradientMachineDestroy(PD_GradiemtMachine machine);

int PDInit(int argc, char** argv);

#ifdef __cplusplus
}
#endif
#endif
