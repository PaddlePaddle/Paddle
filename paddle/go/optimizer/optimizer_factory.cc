#include "optimizer_factory.h"

namespace paddle {
namespace optimizer {

template <class T>
MomentumOptimizer<T>::MomentumOptimizer(const paddle::optimizer_config &config)
    : ParameterOptimizer(config) {
  momentum = config.mometum;
}

}  // namespace optimizer
}  // namespace paddle
