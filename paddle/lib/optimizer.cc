#include "optimizer.h"
#include "optimizer_private.h"

// switch will changed. here show the logic of chosing algorithm id
int32_t paddle_create_optimizer(optimizer* optimizer, optimizer_identifier identifier) {
  if(identifier == 0) {
    /*! \brief TODO: add a new function here?  */ 
    // optimizer->impl = new SGDOptimizer;
  } else if(identifier == 0) {
    // optimizer->impl = new AdamOptimizer;
  }
  return LIB_SUCCESS;
}

int32_t paddle_release_optimizer(optimizer* optimizer) {
  if(optimizer == nullptr)
    return LIB_SUCCESS;
  reinterpret_cast<Coptimizer*>(optimizer)->impl->destory();
  return LIB_SUCCESS;
}

int32_t paddle_update_parameter(optimizer* optimizer, parameter* param, const gradient* grad,
                                paddle_element_type datatype, uint32_t num_bytes, double learning_rate) {
  Tensor<datatype> Parameter(param, num_bytes);
  Tensor<datatype> Gradient(grad, num_bytes);
  /*! \brief real update hook  */ 
  reinterpret_cast<Coptimizer*>(optimizer)->impl->update(Parameter, Gradient, learning_rate);
  return LIB_SUCCESS;
}
