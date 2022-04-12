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

mkdir -p parsed_apis
python parse_api.py \
  --api_yaml_path ./api.yaml \
  --output_path ./parsed_apis/api.parsed.yaml

python parse_api.py \
  --api_yaml_path ./new_api.yaml \
  --output_path ./parsed_apis/new_api.parsed.yaml

python parse_api.py \
  --api_yaml_path ./backward.yaml \
  --output_path ./parsed_apis/backward_api.parsed.yaml \
  --backward

python parse_api.py \
  --api_yaml_path ./new_backward.yaml \
  --output_path ./parsed_apis/new_backward_api.parsed.yaml \
  --backward

echo "Validating api yamls: 
python/paddle/utils/code_gen/api.yaml
python/paddle/utils/code_gen/new_api.yaml
python/paddle/utils/code_gen/backward.yaml
python/paddle/utils/code_gen/new_backward.yaml"
python cross_validate.py \
  --forward_yaml_paths ./parsed_apis/api.parsed.yaml ./parsed_apis/new_api.parsed.yaml \
  --backward_yaml_paths ./parsed_apis/backward_api.parsed.yaml ./parsed_apis/new_backward_api.parsed.yaml

echo "Generating operators and argument mapping functions from api yamls:
paddle/fluid/operators/generated_op.cc.tmp
paddle/phi/ops/compat/generated_sig.cc.tmp"
python generate_op.py \
  --api_yaml_path ./parsed_apis/api.parsed.yaml \
  --backward_api_yaml_path ./parsed_apis/backward_api.parsed.yaml \
  --output_op_path ../../../../paddle/fluid/operators/generated_op.cc.tmp \
  --output_arg_map_path ../../../../paddle/phi/ops/compat/generated_sig.cc.tmp
