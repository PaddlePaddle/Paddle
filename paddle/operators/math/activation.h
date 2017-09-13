#include "paddle/framework/eigen.h"
#include "paddle/framework/tensor.h"

namespace paddle {
namespace operators {
namespace math {

template <typename Place, typename T>
struct sigmoid {
  void operator()(const platform::DeviceContext& deice_context,
                  const framework::Tensor& input, framework::Tensor* output) {
    auto x = framework::EigenVector<T>::Flatten(*output);
    auto y = framework::EigenVector<T>::Flatten(input);
    auto* place = device_context.get_eigen_device<Place>();
    y.device(*place) = 1. / (1. + (-x).exp());
  }
};
}
}
}
