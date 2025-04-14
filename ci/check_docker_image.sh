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


echo Dockerfile: ${docker_image_file}
cd tools/dockerfile
bash ci_dockerfile.sh
docker_md5=`md5sum ${docker_image_file} |awk '{print $1}'`
docker_name=ccr-2vdh3abv-pub.cnc.bj.baidubce.com/ci/paddle:${docker_md5}
echo $docker_name

set +e
  docker pull ${docker_name}
if [ $? -eq 0 ];then
  echo use docker cache
else
  docker build -t $docker_name -f tools/dockerfile/${docker_image_file} .
  echo end docker build
fi
set -e
