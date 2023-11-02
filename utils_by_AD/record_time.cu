// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <iostream>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

#include "stdlib.h"
#include <stdio.h>
#include <dlfcn.h>  // dladdr
#include <sys/time.h>
#include <sys/stat.h>
#include "paddle/extension.h"

std::vector<paddle::Tensor> TimeForward(const paddle::Tensor& x,
                                        const std::string& annotation ) {
  auto out = x.copy_to(x.place(), false);
  auto cu_stream = x.stream();
  auto success = cudaStreamSynchronize(cu_stream);
  using std::chrono::system_clock;
  system_clock::time_point today = system_clock::now();
  auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(today.time_since_epoch()) % (1000 * 60 * 60);
  std::cout << annotation.c_str()<< ": " << milliseconds.count() << std::endl;
  return {out};
}

std::vector<std::vector<int64_t>> TimeInferShape(const std::vector<int64_t>& x_shape) {
    return {x_shape};
}

std::vector<paddle::DataType> TimeInferDtype(const paddle::DataType& x_dtype) {
    return {x_dtype};
}

PD_BUILD_OP(record_time)
    .Inputs({"x"})
    .Attrs({
        "annotation: std::string",
    })
    .Outputs({"out"})
    .SetKernelFn(PD_KERNEL(TimeForward))
    .SetInferShapeFn(PD_INFER_SHAPE(TimeInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(TimeInferDtype));

