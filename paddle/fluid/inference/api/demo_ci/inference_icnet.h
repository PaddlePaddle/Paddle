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

#ifdef _WIN32
#ifdef inference_icnet_EXPORTS
#define API_REFERENCE extern "C" __declspec(dllexport)
#else
#define API_REFERENCE extern "C" __declspec(dllimport)
#endif
#else
#define API_REFERENCE
#endif

API_REFERENCE void* init_predictor(const char* prog_file,
                                   const char* param_file,
                                   const float fraction_of_gpu_memory,
                                   const bool use_gpu, const int device);
API_REFERENCE void predict(void* handle, float* input, const int channel,
                           const int height, const int width, void* output,
                           int& output_length, int batch_size);
API_REFERENCE void predict_file(void* handle, const char* bmp_name,
                                void* output, int& output_length);
API_REFERENCE void destory_predictor(void* handle);
API_REFERENCE void save_image(const char* filename, const void* output,
                              const int output_length);
