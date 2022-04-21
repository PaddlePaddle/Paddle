//
// Created by linxu on 2022/4/21.
//

#include "type_info.h"

int Tinfo::Bits(const DType dtype){
  int bits = elementSize(self->dtype) * 8;
  return THPUtils_packInt64(bits);
}

float Tinfo::Eps(const DType dtype){
    return std::numeric_limits<at::scalar_value_type<scalar_t>::type>::epsilon(dtype));
}

float Tinfo::Min(const DType dtype){
    return std::numeric_limits<at::scalar_value_type<scalar_t>::type>::lowest(dtype));
}

float Tinfo::Max(const DType dtype){
    return std::numeric_limits<at::scalar_value_type<scalar_t>::type>::max(dtype));
}

float Tinfo::Tiny(const DType dtype){
    return std::numeric_limits<at::scalar_value_type<scalar_t>::type>::min(dtype));
}

float Tinfo::Resolution(const DType dtype){
    return std::numeric_limits<at::scalar_value_type<scalar_t>::type>::resolution(dtype));
}