#include "paddle/fluid/operators/npu_utils.h"
#include "paddle/fluid/operators/npu_op_runner.h"

using float16 = paddle::platform::float16;

namespace paddle {
namespace operators {

void alloc_float_status(const paddle::platform::NPUDeviceContext& ctx,
        paddle::framework::Tensor* float_status){
    auto runner = NpuOpRunner("NPUAllocFloatStatus", {}, {*float_status});
    auto stream = ctx.stream();
    runner.Run(stream);
}


bool FoundNanOrInf(const paddle::platform::NPUDeviceContext& ctx, aclrtStream stream, 
        const paddle::framework::Tensor* float_status, Tensor* tmp){
    auto runner_float_status =
        NpuOpRunner("NPUGetFloatStatus", {*float_status}, {*tmp},
                    {{"message", std::string("check_nan_and_inf")}});
    runner_float_status.Run(stream);

    paddle::framework::Tensor sum;
    sum.mutable_data<float>({1}, ctx.GetPlace());
    auto runner_reduce_sum =
        NpuOpRunner("ReduceSumD", {*float_status}, {sum},
                    {{"axes", std::vector<int>{0}}, {"keep_dims", true}});
    runner_reduce_sum.Run(stream);

    std::vector<float> sum_vec;
    TensorToVector(sum, ctx, &sum_vec);
    bool found_inf_data = (sum_vec[0] > 1);

    VLOG(4) << "found_inf_data:" << found_inf_data;
    return found_inf_data;
}


void clear_float_status(const platform::NPUDeviceContext& ctx, 
        Tensor* float_status, Tensor* tmp){
    auto runner_clear_status =
        paddle::operators::NpuOpRunner("NPUClearFloatStatus", {*float_status}, {*tmp});
    runner_clear_status.Run(ctx.stream());
}

};
};


