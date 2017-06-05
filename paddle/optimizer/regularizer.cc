#include "regularizer.h"

namespace paddle {
namespace optimizer {

template <class T>
Regularizer<T>* Regularizer<T>::create(const std::string& config) {
  paddle::OptimizerConfig config;
  Regularizer<T>* r;
  if (config.regularizer_type() == paddle::OptimizerConfig_RegularizerType_L1) {
    r = new L1Regularizer<T>(config);
  } else if (config.regularizer_type() ==
             paddle::OptimizerConfig_RegularizerType_L2) {
    r = new L2Regularizer<T>(config);
    break;
  }
  return r;
}

template class L1Regularizer<float>;
template class L1Regularizer<double>;
template class L2Regularizer<float>;
template class L2Regularizer<double>;

}  // namespace optimizer
}  // namespace paddle
