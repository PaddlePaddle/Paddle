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

cmake_md5=`find ${GITHUB_WORKSPACE} -path './cmake/*' -type f |xargs md5sum|md5sum |awk '{print $1}'`
third_party_md5=`find ${GITHUB_WORKSPACE} -path './third_party/*' -type f |xargs md5sum|md5sum|awk '{print $1}'`
export md5_content=`echo ${cmake_md5} ${third_party_md5} |md5sum|awk '{print $1}'`
xz_dir="$1" && echo "xz_dir=${xz_dir}" >> $GITHUB_ENV
xz_file_tar="${xz_dir}/${md5_content}.tar"
xz_file="${xz_dir}/${md5_content}.tar.xz" && echo "xz_file=${xz_file}" >> $GITHUB_ENV
if [ ! -f "${xz_file}" ];then
    set +e
    wget -q --no-proxy --no-check-certificate https://paddle-github-action.bj.bcebos.com/home/.cache/$2/third_party/${md5_content}.tar.xz; result=$?
    if [ $result -eq 0 ];then
        mkdir -p ${GITHUB_WORKSPACE}/build
        mv ${md5_content}.tar.xz ${xz_dir}
    else
        mkdir -p ${xz_dir}
        echo "update_cached_package=ON" >> $GITHUB_ENV
    fi
else
    mkdir -p ${GITHUB_WORKSPACE}/build
fi
