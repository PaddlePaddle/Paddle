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
/**
 * this group interface called in order : 
 * 1. create optimizer with config
 * 2. set weights
 * 3. update_parameter
 * 4. get_weights
 * 5. release optimizer
 */


/**
 *  @brief create optimizer with proto_config
 *  @param config_proto, optimizer protobuf, see OptimizerConfig.proto in detail
 *  @return return optimizer instance
 */
paddle_optimizer* paddle_create_optimizer(const unsigned char* config_proto,
                                          int config_proto_len);

/**
 *  @brief release optimizer
 *  @param optimizer
 *  @return return exec status
 */
int paddle_release_optimizer(paddle_optimizer* o);

/**
 *  @brief optimizer instance
 *  @param datatype of gradient and parameter
 *  @param gradient, calculate by optimzizer caller.
 *       TODO(zhihong): just pass loss to reduce communicate overhead.
 *                     Project Adam Ms'14 paper for detail
 *  @param num_bytes, gradient size
 *  @return return exec status
 */
int paddle_update_parameter(paddle_optimizer* o,
                            paddle_element_type data_type,
                            const void* gradient,
                            int num_bytes);

/**
 *  @brief optimizer instance
 *  @param data_type datatype of gradient
 *  @param param_buffer, initilized parameter buffer
 *  @param num_bytes, parameter size
 *  @return return exec status
 */
int paddle_optimizer_set_weights(paddle_optimizer* o,
                                 paddle_element_type data_type,
                                 void* param_buffer,
                                 int num_bytes);

/**
 *  @brief optimizer instance
 *  @return return content of parameter buffer in optimizer
 */
void* paddle_optimizer_get_weights(paddle_optimizer* o);

#ifdef __cplusplus
}
#endif
#endif
