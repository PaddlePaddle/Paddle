#include "paddle/fluid/eager/grad_node_info.h"

class GradNodesigmoid : public egr::GradNodeBase {
 public:
  GradNodesigmoid() : egr::GradNodeBase() {}
  GradNodesigmoid(size_t bwd_in_slot_num, size_t bwd_out_slot_num) : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~GradNodesigmoid() override = default;

  virtual std::vector<std::vector<pt::Tensor>> operator()(const std::vector<std::vector<pt::Tensor>>& grads) override;

  // SetX, SetY, ...
   void SetTensorWrapperOut(const pt::Tensor& Out) {
     Out_ = Out;
   }

  // SetAttr0, SetAttr1, ...
   void SetAttruse_cudnn(const bool use_cudnn) {
     use_cudnn_ = use_cudnn;
   }
   void SetAttruse_mkldnn(const bool use_mkldnn) {
     use_mkldnn_ = use_mkldnn;
   }


 private:
   // TensorWrappers
   pt::Tensor Out_;

   // Attribute Members
   bool use_cudnn_ = 0;
   bool use_mkldnn_ = 0;

};