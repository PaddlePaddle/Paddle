#!/bin/bash

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

pr_whl_path=${PADDLE_ROOT}/build/pr_whl/
dev_whl_path=${PADDLE_ROOT}/build/python/dist/
approval_line=$(curl -H "Authorization: token ${GITHUB_API_TOKEN}" https://api.github.com/repos/PaddlePaddle/Paddle/pulls/${GIT_PR_ID}/reviews?per_page=10000)

# Install dev wheel
echo "::group::Generate dev_phi_kernels.json"
pip install --force-reinstall "${dev_whl_path}"/*.whl
python -c 'import paddle;import json;print(json.dumps(paddle.base.core._get_all_register_op_kernels("phi"), indent=4))' > dev_phi_kernels.json
echo "::endgroup::"
# Install pr wheel
echo "::group::Generate pr_phi_kernels.json"
pip install --force-reinstall "${pr_whl_path}"/*.whl
python -c 'import paddle;import json;print(json.dumps(paddle.base.core._get_all_register_op_kernels("phi"), indent=4))' > pr_phi_kernels.json
echo "::endgroup::"

if ! python "${PADDLE_ROOT}/tools/gpu_kernel_compare.py" dev_phi_kernels.json pr_phi_kernels.json --ignore-data-type-changes; then
  APPROVALS=$(echo "${approval_line}"|python "${PADDLE_ROOT}"/tools/check_pr_approval.py 1 wanghuancoder)
  if [[ "${APPROVALS}" == "FALSE" ]]; then
	echo "**************************************************************"
	echo "Please ensure the added GPU kernel supports large tensors, defined as those with a number of elements (numel) greater than 2^31 - 1. In the PR description, please describe how you tested this scenario and validated the data accuracy."
	echo "You must have one RD (wanghuancoder) approval"
	echo "**************************************************************"
	exit 6
  fi
fi

echo "GPU kernel approval check passed."
