/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/core/compat/op_utils.h"

namespace phi {

KernelSignature BilinearInterpOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("bilinear_interp_v2",
                         {"X", "OutSize", "SizeTensor", "Scale"},
                         {"data_layout",
                          "out_d",
                          "out_h",
                          "out_w",
                          "scale",
                          "interp_method",
                          "align_corners",
                          "align_mode"},
                         {"Out"});
}

KernelSignature NearestInterpOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("nearest_interp_v2",
                         {"X", "OutSize", "SizeTensor", "Scale"},
                         {"data_layout",
                          "out_d",
                          "out_h",
                          "out_w",
                          "scale",
                          "interp_method",
                          "align_corners",
                          "align_mode"},
                         {"Out"});
}
KernelSignature TrilinearInterpOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("trilinear_interp_v2",
                         {"X", "OutSize", "SizeTensor", "Scale"},
                         {"data_layout",
                          "out_d",
                          "out_h",
                          "out_w",
                          "scale",
                          "interp_method",
                          "align_corners",
                          "align_mode"},
                         {"Out"});
}

KernelSignature LinearInterpOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("linear_interp_v2",
                         {"X", "OutSize", "SizeTensor", "Scale"},
                         {"data_layout",
                          "out_d",
                          "out_h",
                          "out_w",
                          "scale",
                          "interp_method",
                          "align_corners",
                          "align_mode"},
                         {"Out"});
}

KernelSignature BicubicInterpOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("bicubic_interp_v2",
                         {"X", "OutSize", "SizeTensor", "Scale"},
                         {"data_layout",
                          "out_d",
                          "out_h",
                          "out_w",
                          "scale",
                          "interp_method",
                          "align_corners",
                          "align_mode"},
                         {"Out"});
}

KernelSignature BilinearInterpGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("bilinear_interp_v2_grad",
                         {"X", "OutSize", "SizeTensor", "Scale", "Out@GRAD"},
                         {"data_layout",
                          "out_d",
                          "out_h",
                          "out_w",
                          "scale",
                          "interp_method",
                          "align_corners",
                          "align_mode"},
                         {"X@GRAD"});
}

KernelSignature NearestInterpGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("nearest_interp_v2_grad",
                         {"X", "OutSize", "SizeTensor", "Scale", "Out@GRAD"},
                         {"data_layout",
                          "out_d",
                          "out_h",
                          "out_w",
                          "scale",
                          "interp_method",
                          "align_corners",
                          "align_mode"},
                         {"X@GRAD"});
}
KernelSignature TrilinearInterpGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("trilinear_interp_v2_grad",
                         {"X", "OutSize", "SizeTensor", "Scale", "Out@GRAD"},
                         {"data_layout",
                          "out_d",
                          "out_h",
                          "out_w",
                          "scale",
                          "interp_method",
                          "align_corners",
                          "align_mode"},
                         {"X@GRAD"});
}

KernelSignature LinearInterpGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("linear_interp_v2_grad",
                         {"X", "OutSize", "SizeTensor", "Scale", "Out@GRAD"},
                         {"data_layout",
                          "out_d",
                          "out_h",
                          "out_w",
                          "scale",
                          "interp_method",
                          "align_corners",
                          "align_mode"},
                         {"X@GRAD"});
}

KernelSignature BicubicInterpGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("bicubic_interp_v2_grad",
                         {"X", "OutSize", "SizeTensor", "Scale", "Out@GRAD"},
                         {"data_layout",
                          "out_d",
                          "out_h",
                          "out_w",
                          "scale",
                          "interp_method",
                          "align_corners",
                          "align_mode"},
                         {"X@GRAD"});
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(bilinear_interp_v2,
                           phi::BilinearInterpOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(nearest_interp_v2,
                           phi::NearestInterpOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(trilinear_interp_v2,
                           phi::TrilinearInterpOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(linear_interp_v2,
                           phi::LinearInterpOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(bicubic_interp_v2,
                           phi::BicubicInterpOpArgumentMapping);

PD_REGISTER_ARG_MAPPING_FN(bilinear_interp_v2_grad,
                           phi::BilinearInterpGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(nearest_interp_v2_grad,
                           phi::NearestInterpGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(trilinear_interp_v2_grad,
                           phi::TrilinearInterpGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(linear_interp_v2_grad,
                           phi::LinearInterpGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(bicubic_interp_v2_grad,
                           phi::BicubicInterpGradOpArgumentMapping);
