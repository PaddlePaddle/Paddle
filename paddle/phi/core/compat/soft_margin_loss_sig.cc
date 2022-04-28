#include "paddle/phi/core/compat/op_utils.h"

namespace phi{
KernelSignature SoftMarginLossGradOpArgumentMapping(const ArgumentMappingContext& ctx){
return KernelSignature("soft_margin_loss_grad",
                        {GradVarName("Out"),"X","Label"},
                        {},
                        {GradVarName("X")});
                        }
}// namespace phi

PD_REGISTER_ARG_MAPPING_FN(soft_margin_loss_grad,phi::SoftMarginLossGradOpArgumentMapping);
