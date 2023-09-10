#include "paddle/phi/core/compat/op_utils.h"

namespace phi {

KernelSignature FcOpArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature("fc",
                         {"Input", "W", "Bias"},
                         {"in_num_col_dims",
                          "activation_type",
                          "use_mkldnn",
                          "padding_weights",
                          "use_quantizer",
                          "mkldnn_data_type",
                          "Scale_in",
                          "Scale_weights",
                          "Scale_out",
                          "force_fp32_output"},
                         {"Out"});
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(fc, phi::FcOpArgumentMapping);
