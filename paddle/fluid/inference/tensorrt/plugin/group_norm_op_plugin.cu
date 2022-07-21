/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/inference/tensorrt/plugin/group_norm_op_plugin.h"

#include "paddle/phi/kernels/group_norm_kernel.h"
#include "paddle/phi/kernels/gpu/group_norm_utils.h"

#include "paddle/phi/common/layout.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {
    using DataLayout = framework::DataLayout;
    template <typename T>
    __global__ void GroupNormForward(const T* x,
                                    const T* mean,
                                    const T* var,
                                    const T* scale,
                                    const T* bias,
                                    int N,
                                    int C,
                                    int W,
                                    int imsize,
                                    int groups,
                                    int group_size,
                                    T epsilon,
                                    T* y,
                                    T* real_var) {
        int gid = blockIdx.y;
        int cid = blockIdx.x;
        int bid = blockIdx.z;
        int H = imsize / W;
        int ccid = gid * group_size + cid;
        if (ccid >= C) return;
        auto ng = bid * groups + gid;
        T x_mean = mean[ng];
        T x_var = var[ng];
        x_var = x_var - x_mean * x_mean;
        T var_inv = rsqrt(x_var + epsilon);
        if (cid == 0 && threadIdx.x == 0) {
            real_var[ng] = x_var;
        }
        for (int imid = threadIdx.x; imid < imsize; imid += blockDim.x) {
            T val;
            int hid, wid;
            int index = (bid * C + ccid) * imsize + imid;
            val = x[index];
            /*
            if (data_layout == DataLayout::kNCHW) {
                val = x[index];
            } else {
                hid = imid / W;
                wid = imid % W;
                val = x[(bid * H + hid) * W * C + wid * C + ccid];
            }
            */
            val = (val - x_mean) * var_inv;

            val *= scale[ccid];
            
            val += bias[ccid];
            y[index] = val;
            /*
            if (data_layout == DataLayout::kNCHW) {
                y[index] = val;
            } else {
                y[(bid * H + hid) * W * C + wid * C + ccid] = val;
            }
            */
        }
    }

nvinfer1::DimsExprs GroupNormPluginDynamic::getOutputDimensions(
        int output_index,
        const nvinfer1::DimsExprs *inputDims,
        int nb_inputs,
        nvinfer1::IExprBuilder &expr_builder) TRT_NOEXCEPT {
  return inputDims[0];
}

bool GroupNormPluginDynamic::supportsFormatCombination(
        int pos,
        const nvinfer1::PluginTensorDesc *in_out,
        int nb_inputs,
        int nb_outputs) TRT_NOEXCEPT {
    PADDLE_ENFORCE_NOT_NULL(
        in_out,
        platform::errors::InvalidArgument(
            "The input of groupnorm plugin shoule not be nullptr."));
    PADDLE_ENFORCE_LT(
        pos,
        nb_inputs + nb_outputs,
        platform::errors::InvalidArgument("The pos(%d) should be less than the "
                                            "num(%d) of the input and the output.",
                                            pos,
                                            nb_inputs + nb_outputs));
    const nvinfer1::PluginTensorDesc &in = in_out[pos];
    if (pos == 0) {
      return (in.type == nvinfer1::DataType::kFLOAT) &&
             (in.format == nvinfer1::TensorFormat::kLINEAR);
    }
    const nvinfer1::PluginTensorDesc &prev = in_out[pos - 1];
    // output
    return in.type == prev.type && in.format == prev.format;
}

nvinfer1::DataType GroupNormPluginDynamic::getOutputDataType(
        int index,
        const nvinfer1::DataType * input_types,
        int nb_inputs) const TRT_NOEXCEPT{
    PADDLE_ENFORCE_EQ(index,
        0,
        platform::errors::InvalidArgument(
            "The groupnorm Plugin only has one input, so the "
            "index value should be 0, but get %d.",
            index));
    return input_types[0];

}

int GroupNormPluginDynamic::enqueue(const nvinfer1::PluginTensorDesc* input_desc,
    const nvinfer1::PluginTensorDesc* output_desc,
    const void* const* inputs,
    //TODO wangbojun check for trt > version 8000
    //TODO void ** outputs should work for trt < 8000
    void * const *outputs,
    void* workspace,
    cudaStream_t stream) TRT_NOEXCEPT {
        const auto &input_dims=input_desc[0].dims;
        int groups=groups_;
        float eps=eps_;

        std::vector<int> input_shape;
        for (int i=0;i<input_dims.nbDims;i++){
            input_shape.push_back(input_dims.d[i]);
        }

        const auto input_ddim = phi::make_ddim(input_shape);
        auto matrix_dim = phi::flatten_to_2d(input_ddim, 1); //check
        int feature_size = static_cast<int>(matrix_dim[1]);
        PADDLE_ENFORCE_EQ(feature_size,
                          scale_.size(),
                          platform::errors::InvalidArgument(
                              "scale's size should be equal to the feature_size in groupnorm,"
                              "but got feature_size:%d, scale's size:%d.",
                              feature_size,
                              scale_.size()));
        PADDLE_ENFORCE_EQ(feature_size,
                          bias_.size(),
                          platform::errors::InvalidArgument(
                              "bias's size should be equal to the feature_size in groupnorm,"
                              "but got feature_size:%d, bias's size:%d.",
                              feature_size,
                              bias_.size()));


        int device_id;
        cudaGetDevice(&device_id);
      
        auto input_type = input_desc[0].type;
        if (input_type == nvinfer1::DataType::kFLOAT) {
            const float *input = reinterpret_cast<const float *>(inputs[0]);
            float *output = static_cast<float *>(outputs[0]);
        
            scale_t.Resize(phi::make_ddim({feature_size}));
            bias_t.Resize(phi::make_ddim({feature_size}));
            
            mean_t.Resize(phi::make_ddim(mean_shape_));
            variance_t.Resize(phi::make_ddim(variance_shape_));
            framework::Tensor temp_variance_t;
            temp_variance_t.Resize(phi::make_ddim(variance_shape_));
            float *scale_d =
                scale_t.mutable_data<float>(platform::CUDAPlace(device_id));
            float *bias_d = bias_t.mutable_data<float>(platform::CUDAPlace(device_id));
            float *mean_d = mean_t.mutable_data<float>(platform::CUDAPlace(device_id));
            float *variance_d =
                variance_t.mutable_data<float>(platform::CUDAPlace(device_id));
            float * temp_variance_d=temp_variance_t.mutable_data<float>(platform::CUDAPlace(device_id));
            cudaMemcpyAsync(scale_d,
                            scale_.data(),
                            sizeof(float) * feature_size,
                            cudaMemcpyHostToDevice,
                            stream);
            cudaMemcpyAsync(bias_d,
                            bias_.data(),
                            sizeof(float) * feature_size,
                            cudaMemcpyHostToDevice,
                            stream);
            const auto input_ddim=phi::make_ddim(input_shape);
            const int C = input_ddim[1];
            const int group_size = C/groups_;
            const int W=input_ddim[input_ddim.size()-1];
            int image_size=1;
            for (int i=2;i<input_ddim.size();++i){
                image_size*=input_ddim[i];
            }
            int block_size=std::min(1024,image_size);
            dim3 grid(group_size,groups_,input_ddim[0]);
            dim3 threads(block_size,1,1);
            using AccT = typename phi::kps::details::MPTypeTrait<float>::Type;
            constexpr int vec_size=sizeof(float4)/sizeof(float);
            int size=group_size*image_size; // group element size
            const int max_num_threads=1024;
            int max_block_size = std::min(size/vec_size,max_num_threads);
            int block_size_nchw=1;
            while(block_size_nchw<max_block_size){
                block_size_nchw*=2;
            }

            block_size_nchw=std::max(block_size_nchw,phi::kps::details::kWarpSize);
            dim3 grids(input_ddim[0]*groups_);
            dim3 blocks(block_size_nchw);
            if (size<vec_size*block_size_nchw){
                phi::ScalarGetMeanAndVarNCHW<float><<<grids, blocks, 0,stream>>>(
                    input, mean_d, temp_variance_d, size);
            } else {
                phi::VectorizedGetMeanAndVarNCHW<float, AccT, vec_size>
                <<<grids,blocks,0,stream>>>(
                    input,mean_d,temp_variance_d,size);
            }
            //int flags =
            //    (scale_data != nullptr) * kHasScale + (bias_data != nullptr) * kHasBias;
            GroupNormForward<float><<<grids,threads,0,stream>>>(
                input,
                mean_d,
                temp_variance_d,
                scale_d,
                bias_d,
                input_ddim[0],
                C,
                W,
                image_size,
                groups_,
                group_size,
                eps_,
                output,
                variance_d);
            /*
            UNROLL_ALL_CASES(3,
              GroupNormForward,
              input,
              mean_d,
              temp_variance_d,
              scale_d,
              bias_d,
              x_dims[0],
              C,
              W,
              imsize,
              groups,
              group_size,
              eps_,
              y,
              variance_d,
              DataLayout::kNCHW); // only support NCHW
              */
        } else {
            // input not float
            PADDLE_THROW(platform::errors::Fatal(
                "The Groupnorm TRT Plugin's only support fp16 input"));        
        }
        return cudaGetLastError() != cudaSuccess;
    }



} // plugin
} // tenssort
} // inference
} // paddle