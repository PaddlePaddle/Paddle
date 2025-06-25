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

BRANCH=$1

MAX_RETRY=5
n=0

while [ "$n" -lt "$MAX_RETRY" ]; do
  git pull upstream "$BRANCH" --no-edit && break
  n=$((n+1))
  echo "git pull failed, retrying in 5 seconds... ($n/$MAX_RETRY)"
  sleep 5
done

if [ "$n" -eq "$MAX_RETRY" ]; then
  echo "Pull failed after $MAX_RETRY retries"
  exit 2
fi
