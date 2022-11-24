// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/core/compat/op_utils.h"

namespace phi {

KernelSignature CropTensorOpArgumentMapping(const ArgumentMappingContext& ctx) {
  if (ctx.InputSize("ShapeTensor") > 0) {
    if (ctx.InputSize("OffsetsTensor") > 0) {
      return KernelSignature(
<<<<<<< HEAD
          "crop_tensor", {"X"}, {"ShapeTensor", "OffsetsTensor"}, {"Out"});
    } else if (ctx.HasInput("Offsets")) {
      return KernelSignature(
          "crop_tensor", {"X"}, {"ShapeTensor", "Offsets"}, {"Out"});
    } else {
      return KernelSignature(
          "crop_tensor", {"X"}, {"ShapeTensor", "offsets"}, {"Out"});
=======
          "crop", {"X"}, {"ShapeTensor", "OffsetsTensor"}, {"Out"});
    } else if (ctx.HasInput("Offsets")) {
      return KernelSignature(
          "crop", {"X"}, {"ShapeTensor", "Offsets"}, {"Out"});
    } else {
      return KernelSignature(
          "crop", {"X"}, {"ShapeTensor", "offsets"}, {"Out"});
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
    }
  } else if (ctx.HasInput("Shape")) {
    if (ctx.InputSize("OffsetsTensor") > 0) {
      return KernelSignature(
<<<<<<< HEAD
          "crop_tensor", {"X"}, {"Shape", "OffsetsTensor"}, {"Out"});
    } else if (ctx.HasInput("Offsets")) {
      return KernelSignature(
          "crop_tensor", {"X"}, {"Shape", "Offsets"}, {"Out"});
    } else {
      return KernelSignature(
          "crop_tensor", {"X"}, {"Shape", "offsets"}, {"Out"});
=======
          "crop", {"X"}, {"Shape", "OffsetsTensor"}, {"Out"});
    } else if (ctx.HasInput("Offsets")) {
      return KernelSignature("crop", {"X"}, {"Shape", "Offsets"}, {"Out"});
    } else {
      return KernelSignature("crop", {"X"}, {"Shape", "offsets"}, {"Out"});
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
    }
  } else {
    if (ctx.InputSize("OffsetsTensor") > 0) {
      return KernelSignature(
<<<<<<< HEAD
          "crop_tensor", {"X"}, {"shape", "OffsetsTensor"}, {"Out"});
    } else if (ctx.HasInput("Offsets")) {
      return KernelSignature(
          "crop_tensor", {"X"}, {"shape", "Offsets"}, {"Out"});
    } else {
      return KernelSignature(
          "crop_tensor", {"X"}, {"shape", "offsets"}, {"Out"});
=======
          "crop", {"X"}, {"shape", "OffsetsTensor"}, {"Out"});
    } else if (ctx.HasInput("Offsets")) {
      return KernelSignature("crop", {"X"}, {"shape", "Offsets"}, {"Out"});
    } else {
      return KernelSignature("crop", {"X"}, {"shape", "offsets"}, {"Out"});
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
    }
  }
}

KernelSignature CropTensorGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  if (ctx.InputSize("OffsetsTensor") > 0) {
    return KernelSignature(
<<<<<<< HEAD
        "crop_tensor_grad", {"X", "Out@GRAD"}, {"OffsetsTensor"}, {"X@GRAD"});
  } else if (ctx.HasInput("Offsets")) {
    return KernelSignature(
        "crop_tensor_grad", {"X", "Out@GRAD"}, {"Offsets"}, {"X@GRAD"});
  } else {
    return KernelSignature(
        "crop_tensor_grad", {"X", "Out@GRAD"}, {"offsets"}, {"X@GRAD"});
=======
        "crop_grad", {"X", "Out@GRAD"}, {"OffsetsTensor"}, {"X@GRAD"});
  } else if (ctx.HasInput("Offsets")) {
    return KernelSignature(
        "crop_grad", {"X", "Out@GRAD"}, {"Offsets"}, {"X@GRAD"});
  } else {
    return KernelSignature(
        "crop_grad", {"X", "Out@GRAD"}, {"offsets"}, {"X@GRAD"});
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
  }
}

}  // namespace phi

<<<<<<< HEAD
=======
PD_REGISTER_BASE_KERNEL_NAME(crop_tensor, crop);
PD_REGISTER_BASE_KERNEL_NAME(crop_tensor_grad, crop_grad);

>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
PD_REGISTER_ARG_MAPPING_FN(crop_tensor, phi::CropTensorOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(crop_tensor_grad,
                           phi::CropTensorGradOpArgumentMapping);
