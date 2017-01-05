#ifndef __PADDLE_PADDLE_CAPI_PADDLECAPI_H_INCLUDED__
#define __PADDLE_PADDLE_CAPI_PADDLECAPI_H_INCLUDED__
#include <stdbool.h>
#include <stdint.h>
#include "config.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  PD_NO_ERROR = 0,
  PD_NULLPTR = 1,
  PD_OUT_OF_RANGE = 2,
  PD_PROTOBUF_ERROR = 3,
  PD_UNDEFINED_ERROR = -1,
} PD_Error;

typedef void* PD_Vector;

int PDVecCreate(PD_Vector* vec, uint64_t size, bool useGpu);

int PDVecDestroy(PD_Vector vec);

int PDVecIsSparse(PD_Vector vec, bool* isSparse);

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

typedef void* PD_GradiemtMachine;

int PDGradientMachineCreateForPredict(PD_GradiemtMachine* machine,
                                      void* modelConfigProtobuf,
                                      int size);

int PDGradientMachineDestroy(PD_GradiemtMachine machine);

int PDInit(int argc, char** argv);

#ifdef __cplusplus
}
#endif
#endif
