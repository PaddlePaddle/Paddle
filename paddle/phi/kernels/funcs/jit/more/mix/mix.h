/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. */

#pragma once

#include <type_traits>

#include "paddle/phi/kernels/funcs/jit/kernel_base.h"

namespace phi {
namespace jit {
namespace more {
namespace mix {
using T = float;

void VSigmoid(const T* x, T* y, int n);
void VTanh(const T* x, T* y, int n);

void LSTMCtHt(lstm_t* step, const lstm_attr_t* attr);
void LSTMC1H1(lstm_t* step, const lstm_attr_t* attr);
void GRUH1(gru_t* step, const gru_attr_t* attr);
void GRUHtPart1(gru_t* step, const gru_attr_t* attr);
void GRUHtPart2(gru_t* step, const gru_attr_t* attr);

#define DECLARE_MORE_KERNEL(name)                                             \
  class name##Kernel : public KernelMore<name##Tuple<T>> {                    \
   public:                                                                    \
    name##Kernel() { this->func = name; }                                     \
    bool CanBeUsed(const typename name##Tuple<T>::attr_type&) const override; \
    const char* ImplType() const override { return "Mixed"; }                 \
  }

// XYN
DECLARE_MORE_KERNEL(VSigmoid);
DECLARE_MORE_KERNEL(VTanh);

// XRN
DECLARE_MORE_KERNEL(LSTMCtHt);
DECLARE_MORE_KERNEL(LSTMC1H1);

DECLARE_MORE_KERNEL(GRUH1);
DECLARE_MORE_KERNEL(GRUHtPart1);
DECLARE_MORE_KERNEL(GRUHtPart2);

#undef DECLARE_MORE_KERNEL

}  // namespace mix
}  // namespace more
}  // namespace jit
}  // namespace phi
