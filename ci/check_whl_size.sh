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

function check_whl_size() {
    if [ ${BRANCH} != 'develop' ];then
        return
    fi

    set +x
    pr_whl_size=`du -m ${PADDLE_ROOT}/build/pr_whl/paddle*.whl|awk '{print $1}'`
    echo "pr_whl_size: ${pr_whl_size}M"

    dev_whl_size=`du -m ${PADDLE_ROOT}/build/python/dist/paddle*.whl|awk '{print $1}'`
    echo "dev_whl_size: ${dev_whl_size}M"

    whldiffSize=`echo $(($pr_whl_size - $dev_whl_size))`
    if [ ${whldiffSize} -gt 10 ]; then
       approval_line=`curl -H "Authorization: token ${GITHUB_TOKEN}" https://api.github.com/repos/PaddlePaddle/Paddle/pulls/${PR_ID}/reviews?per_page=10000`
       APPROVALS=`echo ${approval_line}|python ${PADDLE_ROOT}/tools/check_pr_approval.py 1 zhangbo9674 risemeup1 phlrain`
       echo "current pr ${PR_ID} got approvals: ${APPROVALS}"
       if [ "${APPROVALS}" == "FALSE" ]; then
           echo "=========================================================================================="
           echo "This PR make the release paddlepaddle whl size growth exceeds 10 M."
           echo "Then you must have one RD (zhangbo9674 or risemeup1 or phlrain) approval for this PR\n"
           echo "=========================================================================================="
           exit 6
       fi
    fi
}

check_whl_size
