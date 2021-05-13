#include "paddle/fluid/operators/npu_utils.h"
#include "paddle/fluid/operators/npu_op_runner.h"

using float16 = paddle::platform::float16;

namespace paddle {
namespace operators {

bool FoundNanOrInf(const framework::ExecutionContext& ctx, aclrtStream stream, 
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
    TensorToVector(
        sum, ctx.template device_context<paddle::platform::NPUDeviceContext>(),
        &sum_vec);
    bool found_inf_data = (sum_vec[0] > 1);

    VLOG(4) << "found_inf_data:" << found_inf_data;
    return found_inf_data;
}

};
};


