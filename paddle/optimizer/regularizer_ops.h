#ifndef PADDLE_LIB_REGULARIZER_OPS_H_
#define PADDLE_LIB_REGULARIZER_OPS_H_

namespace paddle {
namespace optimizer {


/*! \brief L1 implement  */
template<class T>
void applyL1(Tensor<T> &parameter,
                    int32_t pass_num,
             double learning_rate) {
  // TODO need to find out how to add pass_num


}

}
}

#endif
