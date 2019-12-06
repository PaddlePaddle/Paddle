#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/sampler.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using Sampler = math::Sampler;
using DDim = framework::DDim;

template <typename DeviceContext, typename T>
class NCESamplerKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto dist_probs = context.Input<Tensor>("CustomDistProbs");
    auto dist_alias = context.Input<Tensor>("CustomDistAlias");
    auto dist_alias_probs = context.Input<Tensor>("CustomDistAliasProbs");
   
    int seed = context.Attr<int>("seed");
    int num_total_classes = context.Attr<int>("num_total_classes");

    PADDLE_ENFORCE_EQ(
        dist_probs->numel(), num_total_classes,
        "ShapeError: The number of elements in Input(CustomDistProbs) "
        "should be equal to the number of total classes. But Received: "
        "Input(CustomDistProbs).numel() = %d, Attr(num_total_classes) "
        "= %d.",
        dist_probs->numel(), num_total_classes);
    PADDLE_ENFORCE_EQ(
        dist_alias->numel(), num_total_classes,
        "ShapeError: The number of elements in Input(CustomDistAlias) "
        "should be equal to the number of total classes. But Received: "
        "Input(CustomDistAlias).numel() = %d, Attr(num_total_classes) "
        "= %d.",
        dist_alias->numel(), num_total_classes);
    PADDLE_ENFORCE_EQ(
        dist_alias_probs->numel(), num_total_classes,
        "ShapeError: The number of elements in Input(CustomDistAliasProbs) "
        "should be equal to the number of total classes. But Received: "
        "Input(CustomDistAliasProbs).numel() = %d, "
        "Attr(num_total_classes) = %d.",
        dist_alias_probs->numel(), num_total_classes);

    const float *probs_data = dist_probs->data<float>();
    const int *alias_data = dist_alias->data<int>();
    const float *alias_probs_data = dist_alias_probs->data<float>();

    Sampler *sampler;
    sampler = new math::CustomSampler(num_total_classes - 1, probs_data,
                                      alias_data, alias_probs_data, seed);
  
    auto output = context.Output<Tensor>("Out");
    auto sample_labels_dims = output->dims();
    int64_t *sample_labels_data =
        output->mutable_data<int64_t>(context.GetPlace());
    
    int64_t index = 0;
    for (int64_t i = 0; i < sample_labels_dims[0]; ++i) {
      for (int64_t j = 0; j < sample_labels_dims[1]; j++) {
        sample_labels_data[index++] = sampler->Sample();
      }
    } 
  }
};
}  // namespace operators
}  // namespace paddle
