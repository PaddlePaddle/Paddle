#ifndef __PADDLE_LIB_C_API_H__
#define __PADDLE_LIB_C_API_H__
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
    PADDLE_ELEMENT_TYPE_INT32   = 0,
    PADDLE_ELEMENT_TYPE_UINT32  = 1,
    PADDLE_ELEMENT_TYPE_INT64   = 2,
    PADDLE_ELEMENT_TYPE_UINT64  = 3,
    PADDLE_ELEMENT_TYPE_FLOAT32 = 4,
    PADDLE_ELEMENT_TYPE_FLOAT64 = 5,
  } paddle_element_type;

  /*! \brief execute status code */ 
  const int32_t LIB_SUCCESS = 0;
  const int32_t LIB_WARNING = 1;
  const int32_t LIB_ERROR   = 2;

  /*! \brief optimizer id, exported to Go */ 
  typedef enum {
    SGD =  0, 
    Adagrad = 1,
    Adam = 2,
    // ...
  } optimizer_identifier;

  typedef void* parameter;
  typedef void* gradient;
  typedef void* optimizer;
/*!
 *  \brief create optimizer function
 *  \param [optimizer] create instance of optimizer
 *  \param [optimizer_identifier] identifier of optimizer method
 *  \return return status code
 */
  int32_t paddle_create_optimizer(optimizer* optimizer, optimizer_identifier identifier);
/*!
 *  \brief release optimizer
 *  \param [optimizer] the optimizer instance 
 *  \return return status code
 */
  int32_t paddle_release_optimizer(optimizer* optimizer);
/*!
 *  \brief this should be thread safe. update parameter with gradient, through the optimizer
 *  \param [param] parameter need to update
 *  \param [grad] gradient caculate by caller
 *  \param [num_bytes] the parameter/gradient size
 *  \param [learning_rate] set learning_rate by the caller
 *  \return return status code
 */
  int32_t paddle_update_parameter(optimizer* optimizer, parameter* param, const gradient* grad,
                                  paddle_element_type type, uint32_t num_bytes, double learning_rate);

#ifdef __cplusplus
}
#endif
#endif
