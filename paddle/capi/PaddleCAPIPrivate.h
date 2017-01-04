#include "PaddleCAPI.h"
#include "paddle/gserver/gradientmachines/GradientMachine.h"
#include "paddle/math/Matrix.h"
#include "paddle/math/Vector.h"
#include "paddle/parameter/Argument.h"
#pragma once

namespace paddle {
namespace capi {

struct CVector {
  VectorPtr vec;
};

struct CMatrix {
  MatrixPtr mat;
};

struct CArguments {
  std::vector<paddle::Argument> args;
};

struct CGradientMachine {
  paddle::GradientMachinePtr machine;
};

template <typename T>
inline T* cast(void* ptr) {
  return reinterpret_cast<T*>(ptr);
}
}
}
