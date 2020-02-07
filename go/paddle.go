package paddle

// #include "paddle/fluid/inference/capi/c_api.h"
import "C"

type DType C.PD_DataType

const {
    FLOAT32 DType = C.PD_FLOAT32
    INT32 DType = C.PD_INT32
    INT64 DType = C.PD_INT64
    UINT8 DType = C.PD_UINT8
    UNKDTYPE DType = C.PD_UNKDTYPE
}

type ZeroCopyTensor struct {
    c *C.PD_ZeroCopyData
}


