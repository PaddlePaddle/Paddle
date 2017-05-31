#include "regularizer.h"

namespace paddle {
namespace optimizer {

template<class T>
Regularizer<T>* Regularizer<T>::create(const std::string& config) {
  paddle::OptimizerConfig config;
  Regularizer<T>* r;
  switch (config.regularizer_type) {
  case paddle::OptimizerConfig_RegularizerType_L1:
    r = new L1Regularizer<T>(config); break;
  case paddle::OptimizerConfig_RegularizerType_L2:
    r = new L2Regularizer<T>(config); break;
}
  return r;
}



// template class L1Regularizer<float>;
// template class Regularizer<double>;

}  // namespace optimizer
}  // namespace paddle

