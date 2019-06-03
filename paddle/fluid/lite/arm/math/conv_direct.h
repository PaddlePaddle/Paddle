// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <cmath>
// #include "paddle/fluid/lite/arm/math/conv_impl_base.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

// // TODO(TJ): move to somewhere else
// template <typename TargetType,
//         PrecisionType PrecisionType,
//         typename Param>
// class ImplBase {
// public:
//     ImplBase() {}
//     virtual ~ImplBase() {}

//     virtual SaberStatus init(const std::vector<Tensor<TargetType>* >& inputs,
//               std::vector<Tensor<TargetType> *>& outputs,
//               Param &param, Context<TargetType > &ctx) {
//       return SaberUnImplError;
//     }

//     virtual SaberStatus create(const std::vector<Tensor<TargetType>* >&
//     inputs,
//                 std::vector<Tensor<TargetType> *>& outputs,
//                 Param &param, Context<TargetType> &ctx) {
//       return SaberUnImplError;
//     }

//     virtual SaberStatus dispatch(const std::vector<Tensor<TargetType>* >&
//     inputs,
//                   std::vector<Tensor<TargetType> *>& outputs,
//                   Param &param) {
//       return SaberUnImplError;
//     }
//     void set_op_name(const char* name){_op_name = name;}
//     const char* get_op_name() { return _op_name.c_str();}

// protected:
//     Param* _param;
//     Context<TargetType>* _ctx;
// };

template <PrecisionType OpDtype>
class DirectConv {
 public:
  typedef typename DataTrait<ARM, OpDtype>::Dtype OpDataType;
  typedef void (*conv_direct_impl)(const float* din, float* dout, int num,
                                   int chout, int hout, int wout, int chin,
                                   int hin, int win, const float* weights,
                                   const float* bias, ConvParam<ARM>& param,
                                   Context<ARM>* ctx);

  typedef void (*conv_direct_int8_impl)(
      const int8_t* din, int32_t* dout, int num, int chout, int hout, int wout,
      int chin, int hin, int win, const int8_t* weights, const int32_t* bias,
      ConvParam<ARM>& param, Context<ARM>* ctx, PrecisionType out_type,
      const float* scale);
  DirectConv() = default;
  ~DirectConv() {}

  virtual bool init(const std::vector<Tensor<ARM>*>& inputs,
                    std::vector<Tensor<ARM>*>& outputs, ConvParam<ARM>& param,
                    Context<ARM>& ctx);

  virtual bool create(const std::vector<Tensor<ARM>*>& inputs,
                      std::vector<Tensor<ARM>*>& outputs, ConvParam<ARM>& param,
                      Context<ARM>& ctx);

  virtual bool dispatch(const std::vector<Tensor<ARM>*>& inputs,
                        std::vector<Tensor<ARM>*>& outputs,
                        ConvParam<ARM>& param);

 private:
  conv_direct_impl _impl{nullptr};
  conv_direct_int8_impl _impl_int8{nullptr};
  bool _is_trans_weights{false};
  Tensor<ARM> _weights_trans;
  std::vector<float> _w_scale;
  Tensor<ARM> _tmp_out;

  Param* _param;
  Context<TargetType>* _ctx;
};

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
