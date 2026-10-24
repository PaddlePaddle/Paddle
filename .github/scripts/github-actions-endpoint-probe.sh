#!/usr/bin/env bash

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

# Probe the network paths used by a self-hosted GitHub Actions runner.
#
# Optional environment variables:
#   PROBE_CONNECT_TIMEOUT  TCP/TLS connect timeout in seconds (default: 10)
#   PROBE_MAX_TIME         Total timeout for each request in seconds (default: 30)
#   PROBE_IPV6             Set to 1 to run additional IPv6 probes
#   PROBE_REPORT_FILE      Report path (default: $GITHUB_WORKSPACE/actions-network-probe.log)
#   BLOB_PROBE_JOB_ID      A completed job whose persisted log is known to exist
#   GITHUB_TOKEN           Token with Actions read permission, supplied by the workflow

set -uo pipefail

CONNECT_TIMEOUT="${PROBE_CONNECT_TIMEOUT:-10}"
MAX_TIME="${PROBE_MAX_TIME:-30}"
PROBE_IPV6="${PROBE_IPV6:-0}"
REPORT_FILE="${PROBE_REPORT_FILE:-${GITHUB_WORKSPACE:-$PWD}/actions-network-probe.log}"
FAILURES=0

ENDPOINTS=(
  "GitHub API|https://api.github.com/zen"
  "Codeload|https://codeload.github.com/_ping"
  "Actions token|https://vstoken.actions.githubusercontent.com/_apis/health"
  "Actions pipelines|https://pipelines.actions.githubusercontent.com/_apis/health"
  "Actions results|https://results-receiver.actions.githubusercontent.com/health"
)

timestamp() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

emit_annotation() {
  local level="$1"
  local title="$2"
  local message="$3"

  if [[ "${GITHUB_ACTIONS:-false}" != "true" ]]; then
    return
  fi

  message="${message//'%'/'%25'}"
  message="${message//$'\r'/'%0D'}"
  message="${message//$'\n'/'%0A'}"
  echo "::${level} title=${title}::${message}"
}

url_host() {
  local value="${1#*://}"
  printf '%s\n' "${value%%/*}"
}

resolve_host() {
  local host="$1"

  echo "DNS records for ${host}:"
  if command -v getent >/dev/null 2>&1; then
    getent ahosts "$host" 2>&1 | awk '!seen[$0]++ { print "  " $0 }' || true
  elif command -v nslookup >/dev/null 2>&1; then
    nslookup "$host" 2>&1 | sed 's/^/  /' || true
  else
    echo "  SKIP: neither getent nor nslookup is installed"
  fi
}

curl_probe() {
  local family="$1"
  local label="$2"
  local url="$3"
  local host
  local result
  local rc

  host="$(url_host "$url")"
  echo
  echo "[$(timestamp)] ${label} (IPv${family})"
  echo "URL: ${url}"
  resolve_host "$host"

  set +e
  result="$(curl "-${family}" --silent --show-error --output /dev/null \
    --connect-timeout "$CONNECT_TIMEOUT" \
    --max-time "$MAX_TIME" \
    --write-out 'http=%{http_code} remote_ip=%{remote_ip} dns=%{time_namelookup}s connect=%{time_connect}s tls=%{time_appconnect}s total=%{time_total}s' \
    "$url" 2>&1)"
  rc=$?
  set -e

  echo "curl_exit=${rc} ${result}"
  if (( rc != 0 )); then
    echo "RESULT: FAIL"
    emit_annotation error "${label} IPv${family} unreachable" "${result}"
    FAILURES=$((FAILURES + 1))
  else
    # Any HTTP response proves that DNS, TCP and TLS reached the service. Some
    # unauthenticated health endpoints may intentionally return 4xx.
    echo "RESULT: PASS (transport reachable)"
  fi
}

probe_results_storage_host() {
  local host="$1"
  local result
  local rc

  set +e
  result="$(curl -4 --silent --show-error --output /dev/null \
    --connect-timeout 5 \
    --max-time 10 \
    --write-out 'http=%{http_code} remote_ip=%{remote_ip} dns=%{time_namelookup}s connect=%{time_connect}s tls=%{time_appconnect}s total=%{time_total}s' \
    "https://${host}/" 2>&1)"
  rc=$?
  set -e

  echo "${host}: curl_exit=${rc} ${result}"
  if (( rc != 0 )); then
    emit_annotation error "Azure Blob transport unreachable" "${host}: ${result}"
    return 1
  fi

  emit_annotation notice "Azure Blob transport reachable" "${host}: ${result}"
}

probe_results_storage_accounts() {
  local meta_file
  local probe_dir
  local token="${GITHUB_TOKEN:-}"
  local index=0
  local host
  local curl_args=(
    --fail
    --silent
    --show-error
    --connect-timeout "$CONNECT_TIMEOUT"
    --max-time "$MAX_TIME"
    -H 'Accept: application/vnd.github+json'
    -H 'X-GitHub-Api-Version: 2022-11-28'
  )
  local -a hosts=()
  local -a pids=()

  echo
  echo "[$(timestamp)] GitHub Results Azure Blob account probes"
  meta_file="$(mktemp)"
  probe_dir="$(mktemp -d)"
  if [[ -n "$token" ]]; then
    curl_args+=(-H "Authorization: Bearer ${token}")
  fi

  if ! curl "${curl_args[@]}" https://api.github.com/meta --output "$meta_file"; then
    echo "RESULT: FAIL (unable to retrieve GitHub meta domains)"
    emit_annotation error "GitHub meta domains unavailable" "/meta request failed"
    FAILURES=$((FAILURES + 1))
    rm -rf "$probe_dir"
    rm -f "$meta_file"
    return
  fi

  while IFS= read -r host; do
    hosts+=("$host")
    probe_results_storage_host "$host" > "${probe_dir}/${index}.log" 2>&1 &
    pids+=("$!")
    index=$((index + 1))
  done < <(
    grep -oE 'productionresultssa[0-9]+\.blob\.core\.windows\.net' "$meta_file" \
      | sort -u
  )

  if (( index == 0 )); then
    echo "RESULT: FAIL (no Results Azure Blob domains found in GitHub meta data)"
    emit_annotation error "Azure Blob domain list empty" "No productionresultssa host was found"
    FAILURES=$((FAILURES + 1))
  fi

  for ((index = 0; index < ${#hosts[@]}; index++)); do
    if ! wait "${pids[index]}"; then
      FAILURES=$((FAILURES + 1))
    fi
    cat "${probe_dir}/${index}.log"
  done

  rm -rf "$probe_dir"
  rm -f "$meta_file"
}

probe_persisted_job_log_blob() {
  local job_id="${BLOB_PROBE_JOB_ID:-}"
  local repository="${GITHUB_REPOSITORY:-}"
  local token="${GITHUB_TOKEN:-}"
  local headers
  local api_url
  local location
  local blob_host
  local result
  local rc

  echo
  echo "[$(timestamp)] Persisted job-log Blob probe"

  if [[ -z "$job_id" ]]; then
    echo "RESULT: SKIP (BLOB_PROBE_JOB_ID is not set)"
    return
  fi
  if [[ -z "$repository" || -z "$token" ]]; then
    echo "RESULT: SKIP (GITHUB_REPOSITORY or GITHUB_TOKEN is unavailable)"
    return
  fi

  headers="$(mktemp)"
  api_url="https://api.github.com/repos/${repository}/actions/jobs/${job_id}/logs"

  set +e
  curl --silent --show-error --output /dev/null --dump-header "$headers" \
    --connect-timeout "$CONNECT_TIMEOUT" \
    --max-time "$MAX_TIME" \
    -H 'Accept: application/vnd.github+json' \
    -H "Authorization: Bearer ${token}" \
    -H 'X-GitHub-Api-Version: 2022-11-28' \
    "$api_url"
  rc=$?
  set -e

  if (( rc != 0 )); then
    echo "GitHub log redirect request failed with curl exit ${rc}"
    echo "RESULT: FAIL"
    emit_annotation error "Job log redirect unavailable" "GitHub API curl exit ${rc}"
    FAILURES=$((FAILURES + 1))
    rm -f "$headers"
    return
  fi

  location="$(awk 'BEGIN { IGNORECASE=1 } /^location:/ { sub(/^[^:]*:[[:space:]]*/, ""); sub(/\r$/, ""); print }' "$headers" | tail -n 1)"
  rm -f "$headers"

  if [[ -z "$location" ]]; then
    echo "No redirect URL was returned. Confirm that job ${job_id} exists and its log Blob is present."
    echo "RESULT: FAIL"
    emit_annotation error "Job log Blob redirect missing" "Job ${job_id} returned no Location header"
    FAILURES=$((FAILURES + 1))
    return
  fi

  blob_host="$(url_host "$location")"
  echo "Redirect host: ${blob_host}"
  echo "The signed redirect URL is intentionally not printed."
  resolve_host "$blob_host"

  set +e
  result="$(curl -4 --head --silent --show-error --output /dev/null \
    --connect-timeout "$CONNECT_TIMEOUT" \
    --max-time "$MAX_TIME" \
    --write-out 'http=%{http_code} remote_ip=%{remote_ip} dns=%{time_namelookup}s connect=%{time_connect}s tls=%{time_appconnect}s total=%{time_total}s' \
    "$location" 2>&1)"
  rc=$?
  set -e

  echo "curl_exit=${rc} ${result}"
  if (( rc != 0 )); then
    echo "RESULT: FAIL"
    emit_annotation error "Signed job log Blob unreachable" "${blob_host}: ${result}"
    FAILURES=$((FAILURES + 1))
  else
    echo "RESULT: PASS (GitHub log Blob reachable)"
    emit_annotation notice "Signed job log Blob reachable" "${blob_host}: ${result}"
  fi
}

main() {
  local entry
  local label
  local url

  echo "GitHub Actions network probe"
  echo "Started: $(timestamp)"
  echo "Runner: ${RUNNER_NAME:-unknown}"
  echo "Repository: ${GITHUB_REPOSITORY:-unknown}"
  echo "Run: ${GITHUB_RUN_ID:-unknown}, attempt: ${GITHUB_RUN_ATTEMPT:-unknown}"
  echo "Connect timeout: ${CONNECT_TIMEOUT}s; request timeout: ${MAX_TIME}s"

  for entry in "${ENDPOINTS[@]}"; do
    label="${entry%%|*}"
    url="${entry#*|}"
    curl_probe 4 "$label" "$url"
    if [[ "$PROBE_IPV6" == "1" ]]; then
      curl_probe 6 "$label" "$url"
    fi
  done

  probe_results_storage_accounts
  probe_persisted_job_log_blob

  echo
  echo "Finished: $(timestamp)"
  echo "Transport failures: ${FAILURES}"
  echo "Report: ${REPORT_FILE}"
  echo "The workflow's upload-artifact step separately tests a real Blob upload."

  if (( FAILURES > 0 )); then
    return 1
  fi
}

mkdir -p "$(dirname "$REPORT_FILE")"
set +e
main "$@" 2>&1 | tee "$REPORT_FILE"
status=${PIPESTATUS[0]}
set -e
exit "$status"
