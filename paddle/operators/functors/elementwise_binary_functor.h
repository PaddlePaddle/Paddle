#include "paddle/framework/eigen.h"
#include "paddle/framework/tensor.h"

namespace paddle {
namespace operators {
namespace functors {

template <typename Place, typename T>
struct add {
  void operator()(const platform::DeviceContext& deice_context,
                  const framework::Tensor& input1,
                  const framework::Tensor& input2,
                  framework::Tensor* output) {
    framework::EigenVector<T>::Flatten(*output).device(
        *(device_context.get_eigen_device<Place>())) =
        framework::EigenVector<T>::Flatten(input0) +
        framework::EigenVector<T>::Flatten(input1);
  }
};

template <typename Place, typename T>
struct sub {
  void operator()(const platform::DeviceContext& deice_context,
                  const framework::Tensor& input1,
                  const framework::Tensor& input2,
                  framework::Tensor* output) {
    framework::EigenVector<T>::Flatten(*output).device(
        *(device_context.get_eigen_device<Place>())) =
        framework::EigenVector<T>::Flatten(input0) -
        framework::EigenVector<T>::Flatten(input1);
  }
};

}  // namespace functors
}  // namespace operators
}  // namespace paddle
