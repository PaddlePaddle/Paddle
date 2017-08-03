/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#define HAVE_NEON
#include <arm_neon.h>
#endif

namespace paddle {
namespace neon {

template <int filterSize, int stride>
struct DepthwiseConvKernel{};

struct NaiveDepthwiseConv {
    static void run() {
	}

};

#ifdef HAVE_NEON

template <>
struct DepthwiseConvKernel<3, 1>{
    static void run(){
    }
};

template <>
struct DepthwiseConvKernel<3, 2>{
    static void run(){
    }
};

#endif 

void DepthwiseConvTypeGuide(){

}

} // namespace neon
} // namespace paddle


