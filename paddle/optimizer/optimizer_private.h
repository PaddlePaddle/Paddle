#ifndef PADDLE_OPTIMIZER_PRIVATE_H_
#define PADDLE_OPTIMIZER_PRIVATE_H_

// #include "math/Tensor.h" 
#include "optimizer.h"
#include "Tensor.h"


struct paddle_optimizer {
  /*! \brief optmizer in C++ side */ 
  paddle::optimizer::ParameterOptimzier *impl;
};
