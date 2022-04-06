# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

python parse_api.py \
  --api_yaml_path ./api.yaml \
  --output_path ./temp/api.parsed.yaml

python parse_api.py \
  --api_yaml_path ./backward.yaml \
  --output_path ./temp/backward_api.parsed.yaml \
  --backward

python cross_validate.py \
  --forward_yaml_path ./temp/api.parsed.yaml \
  --backward_yaml_path ./temp/backward_api.parsed.yaml

python generate_op.py \
  --api_yaml_path ./temp/api.parsed.yaml \
  --backward_api_yaml_path ./temp/backward_api.parsed.yaml \
  --output_dir ./temp
