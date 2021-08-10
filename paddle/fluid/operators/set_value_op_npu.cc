/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */



#include "paddle/fluid/operators/mean_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/assign_value_op.h"
#include "paddle/fluid/operators/slice_utils.h"
#include "paddle/fluid/operators/utils.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace operators {
template <typename T>
void print_matrix(const framework::ExecutionContext& ctx, const Tensor& t) {
  std::vector<T> bad_out_data(t.numel());
  framework::TensorToVector<T>(
      t, ctx.template device_context<paddle::platform::NPUDeviceContext>(),
      &bad_out_data);
  std::string ret = "";
  int cols = t.dims()[t.dims().size() - 1] ; 
  int rows = t.dims()[t.dims().size() - 2] ; 
  for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
	  ret += std::to_string(bad_out_data[i*cols+j]) + " ";
      }
      ret += "\n" ; 
  }
  VLOG(4) << t.dims() << "DATA: \n" << ret << std::endl;
}

template <typename T>
void print_vector(const std::vector<T> & in) {
    std::string ret = "" ; 
    for (unsigned long int i=0;i<in.size();++i) {
    	ret += std::to_string(in[i]) + " " ; 
    }
    VLOG(4) << ret << "\n" << std::endl ; 
}

std::vector<float> get_range(int n) {
    std::vector<float> res ; 
    for (int i=0;i<n;++i) {
	res.push_back(i) ; 
    }
    return res ; 
}

template <typename DeviceContext, typename T>
class SetValueNPUKernel : public framework::OpKernel<T> {
private:
  using vec64 = std::vector<int64_t> ; 
  using vec32 = std::vector<int32_t> ; 
  using vec_vec64 = std::vector<std::vector<int64_t> > ;
  inline void GetNPUStartEndSteps(vec_vec64 & output, vec64 & start, vec64 & end, vec64 & steps, vec64 & axes, framework::DDim in_dim)const {
    int rank = in_dim.size() ; 
    for (int i=0;i<rank;++i) {
    	int  axis_size = in_dim[i] ; 
    	auto iter = find(axes.begin(), axes.end(), i) ; 
	if (iter != axes.end()) {
	    // find 
	    int idx = iter - axes.begin() ;  
	    output[0].push_back(start[idx]) ; // set as the same as raw input
	    output[1].push_back(end[idx]) ;   
	    output[2].push_back(steps[idx]) ; 
	} else {
	    output[0].push_back(0) ;  // begin 0 
	    output[1].push_back(axis_size+1) ; // end = last one
	    output[2].push_back(1) ; // step = 1
	}
    }
  }

  inline std::vector<int> MininumPadNumberMakeSureLastDimGT8(vec_vec64 & npu_slice)const {
    int min_value = 32 / sizeof(T) ;  // 16 for float16 , 8 for float32
    int rank = npu_slice[0].size() ; 
    int last_dim_start = npu_slice[0][rank-1] ; 
    int last_dim_end = npu_slice[1][rank-1] ; 
    int last_dim_step = npu_slice[2][rank-1] ; 
    int min_end = last_dim_start + last_dim_step * min_value ; 
    int raw_last_dim_len = (last_dim_end - last_dim_start) / last_dim_step ; 
    // if > 0, set the end ;
    if (min_end - last_dim_end > 0) {
    	npu_slice[1][rank-1] = min_end ; // enforce the last dim > 8
    }
    return std::vector<int>({std::max(0, min_end - last_dim_end), min_value - raw_last_dim_len}) ; 
  }

  inline void PadTensor(const framework::ExecutionContext* ctx, Tensor * output, const Tensor * input, int pad_number)const {
    VLOG(4) << "start pad tensor function, which call the NPU operator PadD" << std::endl ; 
    auto in_dims = input->dims() ; 
    int rank = in_dims.size() ; 
    in_dims[rank-1] += pad_number ;  // output dim
    output->mutable_data<T>(in_dims, ctx->GetPlace()) ; 
    std::vector<std::vector<long int>> pad_attr(rank)  ; 
    for (int i=0;i<rank;++i) pad_attr[i] = std::vector<long int>(2, 0) ; 
    pad_attr[rank-1] = std::vector<long int>({0, pad_number}) ; 
    framework::NPUAttributeMap attr ;
    attr["paddings"] = pad_attr ; 
    auto stream =
        ctx->template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    NpuOpRunner("PadD", {*input}, {*output}, attr).Run(stream);
    VLOG(4) << "PAD TENSOR\n" ; 
    print_matrix<T>(*ctx, *output) ; 
  }

  inline void CropTensor(const framework::ExecutionContext* ctx, Tensor * output, const Tensor * input) const{
    VLOG(4) << "start crop tensor to final shape" << std::endl ; 
    VLOG(4) << "expect crop dims is:" << output->dims() << std::endl ;
    VLOG(4) << "input  crop dims is:" << input->dims() << std::endl ;
    int rank = input->dims().size() ; 
    framework::NPUAttributeMap attr ;
    attr["axis"] = 0 ; 
    attr["offsets"] = std::vector<int>(rank, 0) ; 
    auto stream =
        ctx->template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    NpuOpRunner("Crop", {*input, *output}, {*output}, attr).Run(stream); 
  }
  	
  inline void SliceAssignNPU(const framework::ExecutionContext* ctx, Tensor * output, const Tensor * lefthand, const Tensor * righthand, vec64 & start, vec64 & end, vec64 & steps, vec64 & axes)const {
    // must ensure lefthand and righthand have the same shape
    // not support steps < 0
    // output must have the same type and shape of lefthand. 
    VLOG(4) << "start function SliceAssignND" << std::endl ; 
    auto stream =
        ctx->template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    for (long unsigned int i=0;i<steps.size();++i) {
    	PADDLE_ENFORCE_GT(steps[i], 0, platform::errors::InvalidArgument(
           "currently NPU set_value operator don't support negative steps"
	)); 
    }
    VLOG(4) << "checkpoint 0 " << std::endl ; 
    vec_vec64 npu_slice(3) ; 
    GetNPUStartEndSteps(npu_slice, start, end, steps, axes, lefthand->dims());
    VLOG(4) << "checkpoint 1 " << std::endl ; 
    auto tile_numbers = MininumPadNumberMakeSureLastDimGT8(npu_slice) ; 
    VLOG(4) << "checkpoint 1 " << std::endl ; 
    int lefthand_tile_number = tile_numbers[0] ; 
    int righthand_tile_number = tile_numbers[1] ; 

    VLOG(4) << "tile number is : " << lefthand_tile_number << " " << righthand_tile_number << std::endl ; 

    Tensor tiled_left, tiled_right, before_crop_output ; 
    Tensor * saved_output = output ;
    if (lefthand_tile_number > 0) {
    	PadTensor(ctx, & tiled_left, lefthand, lefthand_tile_number) ; 
	lefthand = & tiled_left ;  
    	PadTensor(ctx, & tiled_right, righthand, righthand_tile_number) ; 
	righthand = & tiled_right ;  
	// output have different shape, so use a tmp variable before_crop_output;
	auto tmp_dim = output->dims() ; 
	tmp_dim[tmp_dim.size()-1] = 8    ; 
	before_crop_output.mutable_data<T>(tmp_dim, ctx->GetPlace());  
	output = & before_crop_output ; 
    }

    framework::NPUAttributeMap attr_input ;
    attr_input["begin"] = std::vector<int>(npu_slice[0].begin(), npu_slice[0].end()); 
    attr_input["end"] = std::vector<int>(npu_slice[1].begin(), npu_slice[1].end())  ; 
    attr_input["strides"] = std::vector<int>(npu_slice[2].begin(), npu_slice[2].end());
    attr_input["begin_mask"] = 0 ; 
    attr_input["end_mask"] = 0 ; 
    attr_input["ellipsis_mask"] = 0 ; 
    attr_input["new_axis_mask"] = 0 ; 
    attr_input["shrink_axis_mask"] = 0 ; 
    NpuOpRunner("StridedSliceAssignD", {*lefthand, *righthand}, {*output}, attr_input).Run(stream);
    if (lefthand_tile_number > 0){
	CropTensor(ctx, saved_output, output) ; 
    }
  }
  inline void ConstructValueTensorFromConstant() {
    return ; 
  }
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    VLOG(2) << "Start Npu Kernel" << std::endl ;
    auto* in = ctx.Input<framework::LoDTensor>("Input");
    auto* out = ctx.Output<framework::LoDTensor>("Out");
    auto* value_tensor = ctx.Input<framework::LoDTensor>("ValueTensor");
    auto starts_tensor_list =
        ctx.MultiInput<framework::Tensor>("StartsTensorList");
    auto ends_tensor_list = ctx.MultiInput<framework::Tensor>("EndsTensorList");
    auto steps_tensor_list =
        ctx.MultiInput<framework::Tensor>("StepsTensorList");
    auto axes = ctx.Attr<std::vector<int64_t>>("axes");
    auto starts = ctx.Attr<std::vector<int64_t>>("starts");
    auto ends = ctx.Attr<std::vector<int64_t>>("ends");
    auto steps = ctx.Attr<std::vector<int64_t>>("steps");
    auto shape = ctx.Attr<std::vector<int64_t>>("shape");
    auto decrease_axes = ctx.Attr<std::vector<int64_t>>("decrease_axes");
    auto dtype = in->type();
    if (!starts_tensor_list.empty()) {
      starts = GetDataFromTensorList<int64_t>(starts_tensor_list);
    }
    if (!ends_tensor_list.empty()) {
      ends = GetDataFromTensorList<int64_t>(ends_tensor_list);
    }
    if (!steps_tensor_list.empty()) {
      steps = GetDataFromTensorList<int64_t>(steps_tensor_list);
    }
    auto in_dims = in->dims();
    CheckAndUpdateSliceAttrs(in_dims, axes, &starts, &ends, &steps);
    auto slice_dims = GetSliceDims(in_dims, axes, starts, ends, &steps);
    auto decrease_slice_dims = GetDecreasedDims(slice_dims, decrease_axes);
    auto place = ctx.GetPlace();

    VLOG(4) << "slice dims:" << slice_dims << std::endl ; 
    VLOG(4) << "decrease slice dims:" << decrease_slice_dims << std::endl ; 

    // Following is not copyed from CPU kernel.
    Tensor slice_tensor(dtype) ;
    slice_tensor.mutable_data<T>(slice_dims, place);

    std::vector<float>val_vec = get_range(16) ;
    // suppose value tensor is not a constant

    print_vector<long int>(starts) ; 
    print_vector<long int>(ends) ; 
    print_vector<long int>(steps) ; 
    print_vector<long int>(axes) ; 

    print_matrix<T>(ctx, *in ) ; 
    print_matrix<T>(ctx, *value_tensor ) ; 
    SliceAssignNPU(&ctx, out, in, value_tensor, starts, ends, steps, axes) ; 
    print_matrix<T>(ctx, *out ) ; 
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_NPU_KERNEL(
    set_value, ops::SetValueNPUKernel<paddle::platform::NPUDeviceContext, int>,
    ops::SetValueNPUKernel<paddle::platform::NPUDeviceContext, float>)
