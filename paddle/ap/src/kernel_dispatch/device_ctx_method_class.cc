// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/ap/include/kernel_dispatch/device_ctx_method_class.h"
#include "paddle/ap/include/axpr/value.h"

namespace ap::kernel_dispatch {

struct DeviceCtxMethodClass {
  using Self = DeviceCtx;

  static adt::Result<axpr::Value> ToString(
      const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    const void* ptr = self.__adt_rc_shared_ptr_raw_ptr();
    std::ostringstream ss;
    ss << "<DeviceCtx object at " << ptr << ">";
    return ss.str();
  }

  static adt::Result<axpr::Value> GetStreamAddrAsVoidPtr(
      const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    ADT_CHECK(args.size() == 0) << adt::errors::TypeError{
        std::string() +
        "DeviceCtx.get_stream_addr_as_void_ptr() takes 0 arguments, but " +
        std::to_string(args.size()) + " were given"};
    ADT_LET_CONST_REF(void_ptr, self.shared_ptr()->GetStreamAddrAsVoidPtr());
    return void_ptr;
  }
};

axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>> GetDeviceCtxClass() {
  using Impl = DeviceCtxMethodClass;
  static auto cls(
      axpr::MakeBuiltinClass<axpr::Value>("DeviceCtx", [&](const auto& Yield) {
        Yield("__str__", &Impl::ToString);
        Yield("get_stream_addr_as_void_ptr", &Impl::GetStreamAddrAsVoidPtr);
      }));
  using Self = typename Impl::Self;
  return axpr::MakeGlobalNaiveClassOps<Self>(cls);
}

}  // namespace ap::kernel_dispatch
