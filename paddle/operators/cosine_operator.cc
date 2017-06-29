#include "paddle/framework/operator.h"

using namespace paddle::framework;

namespace paddle {
namespace operators {

// Sample Operator implement. Show how to implement a Cosine Operator.
template <typename DeviceContext>
class CosineOp final : public Operator<DeviceContext> {
 public:
  explicit CosineOp(const OpDesc& desc) : Operator<DeviceContext>(desc) {}

  /// init attrs that is needed by this Operator, check the legality here.
  Error InitializeAttributes(const AttributeMap& attrs) {
    AttributeReader reader(attrs);
    if (reader.Contains<float>("scale")) {
      scale_ = reader.Get<float>("scale");
    }
    if (scale_ <= 0.0) {
      return Error("scale of CosineOp must be larger than 0.0, get %f", scale_);
    }
    return Error();
  }

  /// Add the actual calculate logic in this function.
  Error Run(std::vector<Variable*>& inputs, std::vector<Variable*>& outputs,
            DeviceContext* context) const override {
    // TODO(to be implement)
    return Error();
  }

 private:
  /// scale_ has a default value 1
  float scale_{1};
};

}  // namespace operators
}  // namespace paddle