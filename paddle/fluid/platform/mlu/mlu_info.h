#pragma once

#ifdef PADDLE_WITH_MLU

#include "cnrt.h"
#include "cnnl.h"
#include "cn_api.h"

namespace paddle {
namespace platform {

using cnStatus = CNresult;
using cnrtStatus = cnrtRet_t;
using cnnlStatus = cnnlStatus_t;
using mluStream = CNqueue;
using mluCnnlHandle = cnnlHandle_t;                                                                
using mluEventHandle = CNnotifier;
using mluDeviceHandle = CNdev;
using mluDim3 = cnrtDim3_t;

void GetMLUDriverVersion(int* x, int* y, int* z);

}  // namespace platform
}  // namespace paddle

#endif