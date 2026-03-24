#!/bin/bash

# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

echo "🔍 [PR Skip Check] Starting..."

# Parse input parameters
IFS=',' read -ra IGNORE_PATHS <<< "${IGNORE_PATHS:-skill/**}"
BASE_REF="${BASE_REF:-$GITHUB_BASE_REF}"

echo "📋 Event type: $GITHUB_EVENT_NAME"
echo "📋 Ignore path patterns:"
printf '  - %s\n' "${IGNORE_PATHS[@]}"

# Initialize output
IS_PR="false"
SHOULD_SKIP="false"
CHANGED_FILES=""
NON_IGNORED_FILES=""
TOTAL_COUNT=0

# Check if PR event
if [ "$GITHUB_EVENT_NAME" = "pull_request" ]; then
    IS_PR="true"
    echo "🔄 Detected PR event, running skip check"

    # Get PR changed files
    get_pr_changed_files() {
        local files=""

        if [ -n "$BASE_REF" ]; then
            echo "📌 Base branch: $BASE_REF" >&2
            files=$(git diff --name-only "origin/$BASE_REF...HEAD" 2>/dev/null)
        else
            echo "⚠️ No base branch specified, using HEAD^..HEAD" >&2
            files=$(git diff --name-only HEAD^ HEAD 2>/dev/null || echo "")
        fi

        echo "$files" | grep -v '^$' || echo ""
    }

    CHANGED_FILES=$(get_pr_changed_files)

    if [ -n "$CHANGED_FILES" ]; then
        echo "📋 PR changed files:"
        echo "$CHANGED_FILES" | sed 's/^/  - /'

        # Count total
        TOTAL_COUNT=$(echo "$CHANGED_FILES" | wc -l | xargs)

        # Check if files match ignore patterns
        check_files() {
            local non_ignored=""

            echo "🔍 [DEBUG] Starting to check files..." >&2

            while IFS= read -r file; do
                [ -z "$file" ] && continue
                echo "🔍 [DEBUG] Checking file: '$file'" >&2

                is_ignored=false
                echo "🔍 [DEBUG] IGNORE_PATHS length: ${#IGNORE_PATHS[@]}" >&2
                for pattern in "${IGNORE_PATHS[@]}"; do
                    pattern=$(echo "$pattern" | xargs)
                    echo "🔍 [DEBUG] Original pattern: '$pattern'" >&2
                    if [ -n "$pattern" ]; then
                        # Replace ** with * for bash pattern matching
                        bash_pattern="${pattern//\**/*}"
                        echo "🔍 [DEBUG] bash_pattern: '$bash_pattern'" >&2
                        # Debug: show pattern matching
                        if [[ "$file" == $bash_pattern ]]; then
                            is_ignored=true
                            echo "  ✅ Ignored: $file (pattern: $pattern -> $bash_pattern)" >&2
                            break
                        else
                            echo "  🔍 Check: $file (pattern: $pattern -> $bash_pattern) - no match" >&2
                        fi
                    fi
                done

                if [ "$is_ignored" = false ]; then
                    echo "  ❌ Not ignored: $file" >&2
                    if [ -z "$non_ignored" ]; then
                        non_ignored="$file"
                    else
                        non_ignored="$non_ignored"$'\n'"$file"
                    fi
                fi
            done <<< "$CHANGED_FILES"

            echo "$non_ignored"
        }

        NON_IGNORED_FILES=$(check_files)

        echo "📋 Non-ignored files:"
        echo "$NON_IGNORED_FILES" | sed 's/^/  - /'

        # Determine if should skip
        if [ -z "$NON_IGNORED_FILES" ]; then
            SHOULD_SKIP="true"
            echo "✅ All PR changes are in ignored paths, can skip CI"
        else
            SHOULD_SKIP="false"
            echo "❌ Some files not in ignored paths, continue CI"
        fi
    else
        echo "📭 No changed files detected in PR"
        SHOULD_SKIP="false"
    fi
else
    # Non-PR event (push, workflow_dispatch, etc.)
    echo "🔄 not a PR event ($GITHUB_EVENT_NAME), run CI"
    SHOULD_SKIP="false"
fi

# Set output variables
echo "should-skip=$SHOULD_SKIP" >> $GITHUB_OUTPUT
echo "is-pr=$IS_PR" >> $GITHUB_OUTPUT
echo "total-count=$TOTAL_COUNT" >> $GITHUB_OUTPUT

# Handle multiline output (changed-files)
if [ -n "$CHANGED_FILES" ]; then
    echo 'changed-files<<EOF' >> $GITHUB_OUTPUT
    echo "$CHANGED_FILES" >> $GITHUB_OUTPUT
    echo 'EOF' >> $GITHUB_OUTPUT
else
    echo "changed-files=" >> $GITHUB_OUTPUT
fi

# Handle multiline output (non-ignored-files)
if [ -n "$NON_IGNORED_FILES" ]; then
    echo 'non-ignored-files<<EOF' >> $GITHUB_OUTPUT
    echo "$NON_IGNORED_FILES" >> $GITHUB_OUTPUT
    echo 'EOF' >> $GITHUB_OUTPUT
else
    echo "non-ignored-files=" >> $GITHUB_OUTPUT
fi

echo "📊 total=$TOTAL_COUNT, can skip=$SHOULD_SKIP"
echo "✅ [PR Skip Check] finished"
