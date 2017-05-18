#ifndef __PADDLE_LIB_REGULARIZER_H
#define __PADDLE_LIB_REGULARIZER_H


/*! \brief regularizer for L1, L2 */

class Regularizer {
public:
/*!
 *  \brief update interface 
 *  \param parameter need to update 
 *  \param pass_num, caller pass the pass_num to regularizer
 *  \return void
 */
  virtual void update(Tensor<T> parameter,
                      int32_t pass_num,
                      double learning_rate) const = 0;   
  virtual ~Regularizer() {}
  
private:
};

class L1LrRegularizer : public Regularizer {
public:
  virtual void update(Tensor<T> parameter,
                      int32_t pass_num,
                      double learning_rate) {
    applyL1(parameter, pass_num, learning_rate);
  }
private:
  
};


#endif
