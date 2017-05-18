#ifndef __PADDLE_OPTIMIZER_PRIVATE_H__
#define __PADDLE_OPTIMIZER_PRIVATE_H__

// #include "math/Tensor.h" 
#include "optimizer.h"
#include "Tensor.h"


struct Coptimizer {
  /*! \brief optmizer in C++ side */ 
  ParameterOptimzier *impl;
};
