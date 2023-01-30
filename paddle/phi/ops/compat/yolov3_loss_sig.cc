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

KernelSignature Yolov3LossOpArgumentMapping(const ArgumentMappingContext& ctx) {
<<<<<<< HEAD
  return KernelSignature("yolo_loss",
=======
  return KernelSignature("yolov3_loss",
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                         {"X", "GTBox", "GTLabel", "GTScore"},
                         {"anchors",
                          "anchor_mask",
                          "class_num",
                          "ignore_thresh",
                          "downsample_ratio",
                          "use_label_smooth",
                          "scale_x_y"},
                         {"Loss", "ObjectnessMask", "GTMatchMask"});
}

KernelSignature Yolov3LossGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature(
<<<<<<< HEAD
      "yolo_loss_grad",
=======
      "yolov3_loss_grad",
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
      {"X",
       "GTBox",
       "GTLabel",
       "GTScore",
       "ObjectnessMask",
       "GTMatchMask",
       "Loss@GRAD"},
      {"anchors",
       "anchor_mask",
       "class_num",
       "ignore_thresh",
       "downsample_ratio",
       "use_label_smooth",
       "scale_x_y"},
      {"X@GRAD", "GTBox@GRAD", "GTLabel@GRAD", "GTScore@GRAD"});
}
}  // namespace phi

<<<<<<< HEAD
PD_REGISTER_BASE_KERNEL_NAME(yolov3_loss, yolo_loss);
PD_REGISTER_BASE_KERNEL_NAME(yolov3_loss_grad, yolo_loss_grad);

=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
PD_REGISTER_ARG_MAPPING_FN(yolov3_loss, phi::Yolov3LossOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(yolov3_loss_grad,
                           phi::Yolov3LossGradOpArgumentMapping);
