# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cd ${work_dir}/build
find ./ -type f -size +200M | xargs du -lh
rm -rf $(find . -name "*.a")
rm -rf $(find . -name "*.o")
rm -rf paddle_inference_install_dir
rm -rf paddle_inference_c_install_dir
rm -rf lib.linux-x86_64-310
find ./ -name "eager_generator" -or -name  "kernel_signature_generator" -or -name "eager_legacy_op_function_generator" | xargs rm -rf
rm -rf ./python/build/lib.linux-x86_64-310/
cd "${work_dir}/build/third_party" && find $(ls | grep -v "dlpack" | grep -v "install" | grep -v "eigen3" | grep -v "gflags") -type f ! -name "*.so" -a ! -name "libdnnl.so*" -delete
