/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "capi.h"
#include "paddle/gserver/gradientmachines/GradientMachine.h"
#include "paddle/math/Matrix.h"
#include "paddle/math/Vector.h"
#include "paddle/parameter/Argument.h"
#pragma once

namespace paddle {
namespace capi {

enum CType { kIVECTOR = 0, kMATRIX, kARGUMENTS, kGRADIENT_MACHINE };

#define STRUCT_HEADER CType type;

struct CHeader {
  STRUCT_HEADER
};

struct CIVector {
  STRUCT_HEADER
  IVectorPtr vec;

  CIVector() : type(kIVECTOR) {}
};

struct CMatrix {
  STRUCT_HEADER
  MatrixPtr mat;

  CMatrix() : type(kMATRIX) {}
};

struct CArguments {
  STRUCT_HEADER
  std::vector<paddle::Argument> args;

  CArguments() : type(kARGUMENTS) {}

  template <typename T>
  paddle_error accessSeqPos(uint64_t ID, uint32_t nestedLevel, T callback) {
    if (ID >= args.size()) return kPD_OUT_OF_RANGE;
    switch (nestedLevel) {
      case 0:
        callback(args[ID].sequenceStartPositions);
        break;
      case 1:
        callback(args[ID].subSequenceStartPositions);
        break;
      default:
        return kPD_OUT_OF_RANGE;
    }
    return kPD_NO_ERROR;
  }
};

struct CGradientMachine {
  STRUCT_HEADER
  paddle::GradientMachinePtr machine;

  CGradientMachine() : type(kGRADIENT_MACHINE) {}
};

template <typename T>
inline T* cast(void* ptr) {
  return reinterpret_cast<T*>(ptr);
}
}  // namespace capi
}  // namespace paddle
