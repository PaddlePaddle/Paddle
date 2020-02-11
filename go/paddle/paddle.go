package paddle

// #include "paddle_c_api.h"
import "C"

import "reflect"

type PaddleDType C.PD_DataType

const {
    FLOAT32 PaddleDType = C.PD_FLOAT32
    INT32 PaddleDType = C.PD_INT32
    INT64 PaddleDType = C.PD_INT64
    UINT8 PaddleDType = C.PD_UINT8
    UNKDTYPE PaddleDType = C.PD_UNKDTYPE
}

type ZeroCopyTensor struct {
    c *C.PD_ZeroCopyTensor
    shape []int
    lod []uint64
}

func NewZeroCopyTensor(value interface{}) *ZeroCopyTensor {
    val := reflect
}

