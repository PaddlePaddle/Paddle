/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/framework/eigen.h"
#include "paddle/platform/operators.h"
#include "paddle/platform/op_registry.h"
#include "paddle/platform/enforce.h"
#include "paddle/platform/ddim.h"
#include <vector>
#include <type_traits>

namespace paddle {
namespace operators {

#define DECLARE_ARG_UTILS(op_type, eigen_op_type, name)													\
class op_type##Utils {																													\
public:																																					\
	inline static constexpr const char* const Name() { return name; } 						\
	inline static constexpr const char* const OpName() { return ##op_type##; }		\
	template <typename Device, typename X, typename Out, typename Tout>						\
	inline static void Run(const Device& d, const X& x, Out& out, int64_t axis) {	\
		static_assert(std::is_integral<Tout>::value, "Tout must be integral type");	\
		out.device(d) = x.##eigen_op_type##(axis).template cast<Tout>();						\
	}																																							\
}

DECLARE_ARG_INFO(ArgMin, argmin, "min");
DECLARE_ARG_INFO(ArgMax, argmax, "max");

template <typename DeviceContext, typename T, typename Tout, typename ArgMinMaxUtils>
class BaseArgMinMaxKernel : public framework::OpKernel<T> {
public:
	void Compute(const framework::ExecutionContext& context) const override {		
		auto& X = detail::Ref(context.Input<framework::Tensor>("X"),
                          "Cannot get input tensor X, variable name = %s",			
                          context.op().Input("X"));
		auto& Out = detail::Ref(context.Output<framework::Tensor>("Out"),						
                            "Cannot get output tensor Out, variable name = %s",
                            context.op().Output("Out"));												
		int64_t axis = context.Attr<int64_t>("axis");
		PADDLE_ENFORE(axis >= 0 && axis < X.dims().size());
		Out.mutable_data<T>(context.GetPlace());
		auto x = framework::EigenVector<T>::Flatten(X);
    auto out = framework::EigenVector<T>::Flatten(Out);
		auto& place = *(context.template device_context<DeviceContext>().eigen_device());
		ArgMinMaxUtils::template Run<decltype((place)), decltype(x), decltype(out), Tout>(place, x, out, axis);
	}
};

template <typename DeviceContext, typename T, typename Tout>
using ArgMinKernel = BaseArgMinMaxKernel<DeviceContext, T, Tout, ArgMinUtils>;

template <typename DeviceContext, typename T, typename Tout>
using ArgMaxKernel = BaseArgMinMaxKernel<DeviceContext, T, Tout, ArgMaxUtils>;


template <typename ArgMinMaxUtils>
class BaseArgMinMaxOpMaker : public framework::OpProtoAndCheckerMaker {
public:
	virtual void Make() override {
		AddInput("X", string::Sprintf("(Tensor), The input tensor of %s op.", ArgMinMaxUtils::OpName()));
		AddOutput("Out", string::Sprintf("(Tensor), The output of %s op.", ArgMinMaxUtils::OpName()));
		AddAttr<int64_t>("axis",
										 "(int64_t, default 0). ",
										 "The axis in which to compute the arg indices.")
			.SetDefault(0)
			.EqualGreaterThan(0);
		AddComment(string::Sprintf(R"DOC(
Computes the indices of the %s elements of the input tensor's element along the provided axis
)DOC", ArgMinMaxUtils::Name()));
	}
};

using ArgMinOpMaker = BaseArgMinMaxOpMaker<ArgMinUtils>;
using ArgMaxOpMaker = BaseArgMinMaxOpMaker<ArgMaxUtils>;

template <typename ArgMinMaxUtils>
class BaseArgMinMaxOp : public framework::OperatorWithKernel {
public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  using Tensor = framework::Tensor;
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   string::Sprintf("Input(X) of %s op should not be null.", ArgMinMaxUtils::OpName()));
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   string::Sprintf("Output(Out) of %s op should not be null.", ArgMinMaxUtils::OpName()));

    auto x_dim = ctx->GetInputDim("X");
    int64_t x_dim_size = x_dim.size();
    std::vector<int64_t> out_dim_vec;
    int64_t axis = ctx->Attrs()->Get<int64_t>("axis");
    out_dim_vec.reserve(x_dim_size - 1);
    for (int64_t i = 0;i < x_dim_size;i ++) {
    	if (i == axis) continue;
    	out_dim_vec.push_back(x_dim[i]);
    }

    ctx->SetOutputDim("Out", make_ddim(out_dim_vec));
  }
};

using ArgMinOp = BaseArgMinMaxOp<ArgMinUtils>;
using ArgMaxOp = BaseArgMinMaxOp<ArgMaxUtils>;
}
}

#define REGISTER_ARG_MINMAX_OP_WITHOUT_GRADIENT(op_type, op_name)						\
	REGISTER_OP_WITHOUT_GRADIENT(op_type, paddle::operators::#op_name#Op, 		\
		paddle::operators::#op_name#OpMaker)

#define REGISTER_ARG_MINMAX_KERNEL_WITH_TYPES(in_type, out_type) 									\
	REGISTER_OP_##LIBRARY_TYPE##_KERNEL(op_type, 																		\
		paddle::operators::##op_name##Kernel<paddle::##LIBRARY_TYPE##DeviceContext, 	\
		in_type, out_type>)

#define REGISTER_ARG_MINMAX_KERNEL(op_type, op_name, LIBRARY_TYPE)	\
	REGISTER_ARG_MINMAX_KERNEL_WITH_TYPES(platform::float16, int64_t);\
	REGISTER_ARG_MINMAX_KERNEL_WITH_TYPES(float, int64_t);						\
	REGISTER_ARG_MINMAX_KERNEL_WITH_TYPES(double, int64_t);						\
	REGISTER_ARG_MINMAX_KERNEL_WITH_TYPES(int64_t, int64_t);					\
	REGISTER_ARG_MINMAX_KERNEL_WITH_TYPES(int32_t, int64_t);					\
	REGISTER_ARG_MINMAX_KERNEL_WITH_TYPES(int16_t, int64_t);					\
	REGISTER_ARG_MINMAX_KERNEL_WITH_TYPES(int8_t, int64_t);						\
	REGISTER_ARG_MINMAX_KERNEL_WITH_TYPES(uint64_t, int64_t);					\
	REGISTER_ARG_MINMAX_KERNEL_WITH_TYPES(uint32_t, int64_t);					\
	REGISTER_ARG_MINMAX_KERNEL_WITH_TYPES(uint16_t, int64_t);					\
	REGISTER_ARG_MINMAX_KERNEL_WITH_TYPES(uint8_t, int64_t)

