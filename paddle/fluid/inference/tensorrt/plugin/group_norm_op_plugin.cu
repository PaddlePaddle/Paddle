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
#include "stdio.h"

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
                                    T* y
                                    //T* real_var
                                    ) {
                                        
        // if(blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0 && threadIdx.x==0){
        //     printf("@@@@trt in group norm kernel \r\n \
        //                 x[0]: %f, mean[0]: %f, var[0]: %f scale[0]: %f, bias[0]: %f,\r\n\
        //                 N: %d, C: %d, W: %d, imsize: %d, groups: %d, group_size: %d \r\n",
        //                 x[0],mean[0],var[0],scale[0],bias[0],N,C,W,imsize,groups,group_size);
        // }                          
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
        /*
        if (cid == 0 && threadIdx.x == 0) {
            real_var[ng] = x_var;
        }
        */
        // if(blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0 && threadIdx.x==0){
        //     printf("@@@@trt in group norm kernel \r\n \
        //     x_mean: %f, x_var %f \r\n",x_mean,x_var);
        //     printf("@@@@trt in group norm kernel, output: \n");
        // }
        for (int imid = threadIdx.x; imid < imsize; imid += blockDim.x) {
            T val;
            int hid, wid;
            int index = (bid * C + ccid) * imsize + imid;
            val = x[index];
            val = (val - x_mean) * var_inv;

            val *= scale[ccid];
            
            val += bias[ccid];
            y[index] = val;
            // if(blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0 && threadIdx.x==0){
            //     printf("ccid: %d, scale: %f, bias: %f",ccid,scale[ccid],bias[ccid]);
            //     printf("x: %f, x_mean: %f, var_inv: %f, index: %d, y: %f", x[index] ,x_mean, var_inv,index,y[index]);
            // }
        }
        // if(blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0 && threadIdx.x==0){
        //     printf("\n");
        // }
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

int GroupNormPluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc* input_desc,
    const nvinfer1::PluginTensorDesc* output_desc,
    const void* const* inputs,
    void * const * outputs,
    void *workspace,
    cudaStream_t stream) TRT_NOEXCEPT {
        const auto &input_dims=input_desc[0].dims;
        int groups=groups_;
        float eps=eps_;

        std::vector<int> input_shape;
        for (int i=0;i<input_dims.nbDims;i++){
            input_shape.push_back(input_dims.d[i]);
        }

        const auto input_ddim = phi::make_ddim(input_shape);
        //VLOG(3)<<"@@@@@ input_ddim :";
        //VLOG(3)<<input_ddim[0]<<" "<<input_ddim[1]<<" "<<input_ddim[2];
        int C = input_shape[1];
        PADDLE_ENFORCE_EQ(C,
                          scale_.size(),
                          platform::errors::InvalidArgument(
                              "scale's size should be equal to the channel number in groupnorm,"
                              "but got feature_size:%d, scale's size:%d.",
                              C,
                              scale_.size()));
        PADDLE_ENFORCE_EQ(C,
                          bias_.size(),
                          platform::errors::InvalidArgument(
                              "bias's size should be equal to the channel number in groupnorm,"
                              "but got feature_size:%d, bias's size:%d.",
                              C,
                              bias_.size()));


        int device_id;
        cudaGetDevice(&device_id);
      
        auto input_type = input_desc[0].type;
        if (input_type == nvinfer1::DataType::kFLOAT) {
            const float *input = static_cast<const float *>(inputs[0]);
            // // for test print block
            // int testInputSize=1*512*144;
            // void * input0_cpu=malloc(sizeof(float)*testInputSize);
            // cudaMemcpy(input0_cpu,(const void *)input,sizeof(float)*testInputSize,cudaMemcpyDeviceToHost);
            // printf("@@@ in enqueue input[0]:\n");
            // for (int i=0;i<testInputSize;i++){
            //     if(i%10==0) printf("\n");
            //     printf(" %f",static_cast<float *>(input0_cpu)[i]);
            // }
            // printf("\n");
            // // for test print block end. 

            float *output = static_cast<float *>(outputs[0]);
        
            scale_t.Resize(phi::make_ddim({C}));
            bias_t.Resize(phi::make_ddim({C}));
            
            VLOG(3)<<"@@@ C : "<<C;
            //VLOG(3)<<"@@@ mean shape: "<<mean_shape_.size();
            //printf("@@@ %d", mean_shape_.size());
            mean_t.Resize(phi::make_ddim(mean_shape_));
            variance_t.Resize(phi::make_ddim(variance_shape_));
            framework::Tensor temp_variance_t;
            temp_variance_t.Resize(phi::make_ddim(variance_shape_));
            float *scale_d =
                scale_t.mutable_data<float>(platform::CUDAPlace(device_id));
            float *bias_d = bias_t.mutable_data<float>(platform::CUDAPlace(device_id));
            float *mean_d = mean_t.mutable_data<float>(platform::CUDAPlace(device_id));
            
            // float *variance_d =
                // variance_t.mutable_data<float>(platform::CUDAPlace(device_id));
            float * temp_variance_d=temp_variance_t.mutable_data<float>(platform::CUDAPlace(device_id));
            cudaMemcpyAsync(scale_d,
                            scale_.data(),
                            sizeof(float) * C,
                            cudaMemcpyHostToDevice,
                            stream);
            cudaMemcpyAsync(bias_d,
                            bias_.data(),
                            sizeof(float) * C,
                            cudaMemcpyHostToDevice,
                            stream);
            VLOG(3)<<"@@@ groups_ "<<groups_;
            const int group_size = C/groups_;
            VLOG(3)<<"@@@ pin";
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
            VLOG(3)<<"@@@ get mean and var";
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
            // printf("trt mean_d");
            int mean_size=1;
            for(int i=0;i<mean_shape_.size();i++){

                mean_size*=mean_shape_[i];
            }
            // // test print block 
            // void * mean_d_cpu=malloc(sizeof(float)*mean_size);
            // cudaMemcpy(mean_d_cpu,(const void *)mean_d,sizeof(float)*mean_size,cudaMemcpyDeviceToHost);
            // for(int i=0;i<mean_size;i++){
            //     if(i%10==0){
            //         printf("\r\n");
            //     }
            //     printf(" %f,",static_cast<float *>(mean_d_cpu)[i]);
            // }
            // printf("\r\n");
            // // test print block end

            VLOG(3)<<"@@@ call group norm forward";
            VLOG(3)<<"@@@ grids_xyz: "<<input_ddim[0]*groups_<<" threads_xyz: "<<block_size_nchw;
            
            // printf("Grid : {%d, %d, %d} blocks. Blocks : {%d, %d, %d} threads.\n",
            // grids.x, grids.y, grids.z, threads.x, threads.y, threads.z);

            GroupNormForward<float><<<grid,threads,0,stream>>>(
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
                output
                //variance_d
                );
            
        } else {
            // input not float
            PADDLE_THROW(platform::errors::Fatal(
                "The Groupnorm TRT Plugin's only support fp32 input"));        
        }
        return cudaGetLastError() != cudaSuccess;
    }



} // plugin
} // tenssort
} // inference
} // paddle