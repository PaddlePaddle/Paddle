#include "optimizer.h"
#include "optimizer_private.h"

int32_t paddle_create_SGDOptimizer(paddle_optimizer* optimizer, double learning_rate) {
  optimizer->impl->create_SGDOptimizer(learning_rate);
  return PADDLE_SUCCESS;
}

int32_t paddle_release_optimizer(paddle_optimizer* optimizer) {
  if(optimizer == nullptr)
    return PADDLE_SUCCESS;
  optimizer->impl->destory();
  return PADDLE_SUCCESS;
}

int32_t paddle_update_parameter(paddle_optimizer* optimizer, parameter* param, const gradient* grad,
                                paddle_element_type datatype, uint32_t num_bytes, double learning_rate) {
  Tensor<datatype> Parameter(param, num_bytes);
  Tensor<datatype> Gradient(grad, num_bytes);
  /*! \brief real update hook  */ 
  optimizer->impl->update(Parameter, Gradient, learning_rate);
  return PADDLE_SUCCESS;
}
