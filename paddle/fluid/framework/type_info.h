#ifndef PADDLE_TYPE_INFO_H
#define PADDLE_TYPE_INFO_H

#include "paddle/infrt/common/dtype.h"

class Tinfo {
int Bits(const DType dtype);
float Eps(const DType dtype);
float Min(const DType dtype);
float Max(const DType dtype);
float Tiny(const DType dtype);
float Resolution(const DType dtype);
}

#endif  // PADDLE_TYPE_INFO_H
