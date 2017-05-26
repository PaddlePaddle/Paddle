#ifndef PADDLE_LIB_OPTIMIZER_H_
#define PADDLE_LIB_OPTIMIZER_H_
#include <stdbool.h>
#include <stdint.h>

/*! \brief optimizer export C API. which will be used in
  Case A, on Trainer (On ParameterServer Client) optimize gradient

  Case B, on ParameterServer side optimize gradient

  To simplify the configuration parsing. optimizer *do not* parse any config
  e.g. learning rate should be calculated by the caller
 */

#ifdef __cplusplus
extern "C" {
#endif
/*! \brief datatypes */
typedef enum {
  PADDLE_ELEMENT_TYPE_INT32 = 0,
  PADDLE_ELEMENT_TYPE_UINT32 = 1,
  PADDLE_ELEMENT_TYPE_INT64 = 2,
  PADDLE_ELEMENT_TYPE_UINT64 = 3,
  PADDLE_ELEMENT_TYPE_FLOAT32 = 4,
  PADDLE_ELEMENT_TYPE_FLOAT64 = 5,
} paddle_element_type;

/*! \brief execute status code */
const int32_t PADDLE_SUCCESS = 0;
const int32_t PADDLE_ERROR = -1;

typedef struct paddle_optimizer paddle_optimizer;

paddle_optimizer* paddle_create_optimizer(const unsigned char* config_proto,
                                          int config_proto_len,
                                          paddle_element_type data_type,
                                          const void* param_buffer,
                                          int num_bytes);

int paddle_release_optimizer(paddle_optimizer* o);

int paddle_update_parameter(paddle_optimizer* o,
                            paddle_element_type datatype,
                            const void* gradient,
                            int num_bytes);

const void* paddle_optimizer_param(paddle_optimizer* o);

// /*!
//  *  \brief create optimizer function
//  *  \param [optimizer] create instance of optimizer
//  *  \param [optimizer_identifier] identifier of optimizer method
//  *  \return return status code
//  */
//   // int32_t paddle_create_XXXOptimizer(paddle_optimizer* optimizer,
//   optimizer_identifier identifier); int32_t
//   paddle_create_SGDOptimizer(paddle_optimizer* optimizer, double
//   learning_rate);
// /*!
//  *  \brief release optimizer
//  *  \param [optimizer] the optimizer instance
//  *  \return return status code
//  */
//   int32_t paddle_release_optimizer(paddle_optimizer* optimizer);
// /*!
//  *  \brief this should be thread safe. update parameter with gradient,
//  through the optimizer *  \param [param] parameter need to update *  \param
//  [grad] gradient caculate by caller *  \param [num_bytes] the
//  parameter/gradient size *  \param [learning_rate] set learning_rate by the
//  caller *  \return return status code
//  */
//   int32_t paddle_update_parameter(paddle_optimizer* optimizer, parameter*
//   param, const gradient* grad,
//                                   paddle_element_type type, uint32_t
//                                   num_bytes, double learning_rate);

#ifdef __cplusplus
}
#endif
#endif
