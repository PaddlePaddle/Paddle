#include "PaddleCAPI.h"
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
}
}
