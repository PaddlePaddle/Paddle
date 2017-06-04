#include "optimizer.h"
#include <string>

#include "parameter_optimizer.h"

template <class T>
struct EnumToType {};

template <class T>
struct TypeToEnum {};

#define MATCH_ENUM_TYPE(TYPE, ENUM)                  \
  template <>                                        \
  struct TypeToEnum<ENUM> {                          \
    static paddle_element_type v() { return ENUM; }; \
    static constexpr TYPE value = ENUM;
}
;
template <>
struct EnumToType<ENUM> {
  typedef TYPE Type;
}

MATCH_ENUM_TYPE(int32_t, PADDLE_ELEMENT_TYPE_INT32);
MATCH_ENUM_TYPE(uint32_t, PADDLE_ELEMENT_TYPE_UINT32);
MATCH_ENUM_TYPE(int64_t, PADDLE_ELEMENT_TYPE_INT64);
MATCH_ENUM_TYPE(uint64_t, PADDLE_ELEMENT_TYPE_UINT64);
MATCH_ENUM_TYPE(float, PADDLE_ELEMENT_TYPE_FLOAT32);
MATCH_ENUM_TYPE(double, PADDLE_ELEMENT_TYPE_FLOAT64);

struct paddle_optimizer {
  /*! \brief optmizer in C++ side */

  paddle::optimizer::ParameterOptimzier* impl;
};

paddle_optimizer* paddle_create_optimizer(const unsigned char* config_proto,
                                          int config_proto_len) {
  paddle_optimizer* optimizer;
  std::string config(config_proto, config_proto + config_proto_len);
  optimizer->impl->create(config_proto);
  return optimizer;
}

int paddle_release_optimizer(paddle_optimizer* o) {
  if (o != nullptr) delete o->impl;
  return PADDLE_SUCCESS;
}

int paddle_update_parameter(paddle_optimizer* o,
                            paddle_element_type data_type,
                            const void* grad_buffer,
                            int num_bytes) {
  auto type = EnumToType<data_type>::Type;
  paddle::Tensor<type> gradient(reinterpret_cast<type*>(grad_buffer),
                                num_bytes);
  o->impl->update(gradient);
  return PADDLE_SUCCESS;
}

int paddle_optimizer_set_weights(paddle_optimizer* o,
                                 paddle_element_type data_type,
                                 void* param_buffer,
                                 int num_bytes) {
  auto type = EnumToType<data_type>::Type;
  paddle::Tensor<type>* param = new paddle::Tensor<type>(
      reinterpret_cast<type*>(param_buffer), num_bytes);
  o->impl->set_weight(param);
  return PADDLE_SUCCESS;
}

void* paddle_optimizer_get_weights(paddle_optimizer* o) {
  void* buffer = (void*)o->impl->get_weight();
  return buffer;
}
