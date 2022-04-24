//
// Created by linxu on 2022/4/21.
//

#include "type_info.h"

int Tinfo::Bits(const DType dtype){
  int bits = elementSize(self->dtype) * 8;
  return THPUtils_packInt64(bits);
}

float Tinfo::Eps(const DType dtype){
    return std::numeric_limits<DType>::epsilon());
}

float Tinfo::Min(const DType dtype){
    return std::numeric_limits<DType>::lowest());
}

float Tinfo::Max(const DType dtype){
    return std::numeric_limits<DType>::max());
}

float Tinfo::Tiny(const DType dtype){
    return std::numeric_limits<DType>::min());
}

float Tinfo::Resolution(const DType dtype){
    return std::numeric_limits<DType>::resolution());
}