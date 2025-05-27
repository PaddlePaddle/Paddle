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

set -e

COMMIT_SHA=$(curl -s -H "Authorization: token $GITHUB_TOKEN" \
  "https://api.github.com/repos/$OWNER/$REPO/pulls/$PR_ID" | jq -r '.head.sha')

echo "Commit SHA: $COMMIT_SHA"

response=$(curl -s -H "Authorization: token $GITHUB_TOKEN" \
  "https://api.github.com/repos/$OWNER/$REPO/actions/runs?head_sha=$COMMIT_SHA&per_page=100")

echo "Response: $response"

run_ids=$(echo "$response" | jq -r '.workflow_runs[].id')

if [ -n "$run_ids" ]; then
  echo "Found run_ids for commit $COMMIT_SHA: $run_ids"

  for run_id in $run_ids; do
    if [ "$JOB_NAME" = "all-failed" ]; then
      echo "Rerunning all failed jobs for run_id: $run_id"

      rerun_response=$(curl -X POST -s -w "%{http_code}" -o /dev/null \
        -H "Accept: application/vnd.github.v3+json" \
        -H "Authorization: Bearer $GITHUB_TOKEN" \
        "https://api.github.com/repos/$OWNER/$REPO/actions/runs/$run_id/rerun-failed-jobs")
      if [ "$rerun_response" -eq 201 ]; then
        echo "Successfully requested rerun for all blocked jobs in run_id: $run_id"
      else
        echo "Failed to request rerun for run_id: $run_id with status code $rerun_response"
      fi

    else
      jobs_response=$(curl -s -H "Authorization: token $GITHUB_TOKEN" \
      "https://api.github.com/repos/$OWNER/$REPO/actions/runs/$run_id/jobs")

      echo "Jobs Response for run_id $run_id: $jobs_response"

      # if [[ "$JOB_NAME" == *"bypass"* ]]; then
        block_jobs=$(echo "$jobs_response" | jq -r --arg job_name "$JOB_NAME" \
        '.jobs[] | select(.name == $job_name) | .id')
      # else
      #   block_jobs=$(echo "$jobs_response" | jq -r --arg job_name "$JOB_NAME" \
      #   '.jobs[] | select(.name == $job_name and .conclusion != "success") | .id')
      # fi

      if [ -n "$block_jobs" ]; then
        echo "Found block jobs for run_id $run_id: $block_jobs"

        for job_id in $block_jobs; do
          echo "Rerunning job_id: $job_id"
          curl -X POST -H "Accept: application/vnd.github.v3+json" \
            -H "Authorization: token $GITHUB_TOKEN" \
            "https://api.github.com/repos/$OWNER/$REPO/actions/jobs/$job_id/rerun"
        done
      else
        echo "No block jobs found for run_id $run_id with name $JOB_NAME."
      fi
    fi
  done
else
  echo "No matching workflow runs found for commit $COMMIT_SHA."
  exit 1
fi
