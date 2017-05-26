#include "optimizer.h"
#include <string>

#include "parameter_optimizer.h"

struct paddle_optimizer {
  /*! \brief optmizer in C++ side */

  paddle::optimizer::ParameterOptimzier* impl;
};

paddle_optimizer* paddle_create_optimizer(const unsigned char* config_proto,
                                          int config_proto_len,
                                          const paddle_element_type data_type,
                                          const void* param_buffer,
                                          int num_bytes) {
  paddle_optimizer* optimizer;
  std::string config(config_proto, config_proto + config_proto_len);
  paddle::Tensor<data_type>* param = new paddle::Tensor<data_type>(
      reinterpret_cast<data_type*>(param_buffer), num_bytes);
  optimizer->impl->create(config_proto, param);
  return optimizer;
}

int paddle_release_optimizer(paddle_optimizer* o) {
  if (o != nullptr) o->impl->destory();
  return PADDLE_SUCCESS;
}

int paddle_update_parameter(paddle_optimizer* o,
                            paddle_element_type data_type,
                            const void* grad_buffer,
                            int num_bytes) {
  paddle::Tensor<data_type> gradient(reinterpret_cast<data_type*>(grad_buffer),
                                     num_bytes);
  o->impl->update(gradient);
  return PADDLE_SUCCESS;
}

const void* paddle_optimizer_param(paddle_optimizer* o) {
  void* buffer = o->impl->get_buffer();
  return buffer;
}

// int32_t paddle_create_SGDOptimizer(paddle_optimizer* optimizer, double
// learning_rate) {
//   optimizer->impl->create_SGDOptimizer(learning_rate);
//   return PADDLE_SUCCESS;
// }

// int32_t paddle_release_optimizer(paddle_optimizer* optimizer) {
//   if (optimizer != nullptr)
//     optimizer->impl->destory();
//   return PADDLE_SUCCESS;
// }

// int32_t paddle_update_parameter(paddle_optimizer* optimizer, parameter*
// param, const gradient* grad,
//                                 paddle_element_type datatype, uint32_t
//                                 num_bytes, double learning_rate) {
//   Tensor<datatype> Parameter(param, num_bytes);
//   Tensor<datatype> Gradient(grad, num_bytes);
//   /*! \brief real update hook  */
//   optimizer->impl->update(Parameter, Gradient, learning_rate);
//   return PADDLE_SUCCESS;
// }
