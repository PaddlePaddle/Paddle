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
for name in `env | grep -E 'PADDLE|ENDPOINT' | awk -F"=" '{print $1}'`; do
unset ${name}
done

export NVSHMEM_DISABLE_P2P="0"
export IP_LIST='127.0.0.1'

export devices=0,1,2,3,4,5,6,7
python -m paddle.distributed.launch \
        --gpus ${devices} \
        --ips ${IP_LIST} \
        test_low_latency_all2all.py
