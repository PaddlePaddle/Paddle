/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */


#ifndef HL_DEVICE_FUNCTIONS_CUH_
#define HL_DEVICE_FUNCTIONS_CUH_

namespace hppl {

static __inline__ __device__ double atomicAdd(double* address, double val) {
    // NOLINTNEXTLINE
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed; // NOLINT

    do {
        assumed = old;
        old = atomicCAS(address_as_ull,
                        assumed,
                        __double_as_longlong(val +
                        __longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
}

}  // namespace hppl

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
using hppl::atomicAdd;
#endif

#endif /* HL_DEVICE_FUNCTIONS_CUH_ */
