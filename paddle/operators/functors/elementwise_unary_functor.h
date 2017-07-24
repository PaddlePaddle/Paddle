#include "paddle/framework/eigen.h"
#include "paddle/framework/tensor.h"

namespace paddle {
namespace operators {
namespace functors {

template <typename Place, typename T>
struct sigmoid {
  void operator()(const platform::DeviceContext& deice_context,
                  const framework::Tensor& input,
                  framework::Tensor* output) {
    framework::EigenVector<T>::Flatten(*output).device(
        *(device_context.get_eigen_device<Place>())) =
        1.0 / (1.0 + (-1.0 * framework::EigenVector<T>::Flatten(input)).exp());
  }
};

}  // namespace functors
}  // namespace operators
}  // namespace paddle
