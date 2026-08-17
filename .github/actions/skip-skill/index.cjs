// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

'use strict';

const fs = require('node:fs');
const path = require('node:path');

const DEFAULT_API_URL = 'https://api.github.com';
const MAX_CHANGED_FILES = 3000;
const PAGE_SIZE = 100;
const REQUEST_ATTEMPTS = 3;
const REQUEST_TIMEOUT_MS = 10_000;

function parseIgnorePatterns(value) {
  if (typeof value !== 'string') {
    return [];
  }
  return value
    .split(',')
    .map((pattern) => pattern.trim())
    .filter(Boolean);
}

function pathMatchesAnyPattern(filePath, patterns) {
  if (typeof filePath !== 'string' || filePath.length === 0) {
    return false;
  }
  return patterns.some((pattern) => path.matchesGlob(filePath, pattern));
}

function collectChangedPaths(files) {
  if (!Array.isArray(files) || files.length === 0) {
    return null;
  }

  const paths = [];
  for (const file of files) {
    if (
      file === null ||
      typeof file !== 'object' ||
      typeof file.filename !== 'string' ||
      file.filename.length === 0
    ) {
      return null;
    }
    paths.push(file.filename);

    if (Object.hasOwn(file, 'previous_filename')) {
      if (
        typeof file.previous_filename !== 'string' ||
        file.previous_filename.length === 0
      ) {
        return null;
      }
      paths.push(file.previous_filename);
    } else if (file.status === 'renamed') {
      return null;
    }
  }
  return paths;
}

function areAllPathsIgnored(files, patterns) {
  if (!Array.isArray(patterns) || patterns.length === 0) {
    return false;
  }
  const changedPaths = collectChangedPaths(files);
  return (
    changedPaths !== null &&
    changedPaths.every((filePath) =>
      pathMatchesAnyPattern(filePath, patterns),
    )
  );
}

function isNetworkError(error) {
  return (
    error instanceof TypeError ||
    error?.name === 'AbortError' ||
    error?.name === 'TimeoutError'
  );
}

function retryDelay(attempt) {
  return new Promise((resolve) => setTimeout(resolve, attempt * 250));
}

async function githubApiRequest(
  url,
  {
    token,
    fetchImpl = globalThis.fetch,
    attempts = REQUEST_ATTEMPTS,
    timeoutMs = REQUEST_TIMEOUT_MS,
    sleep = retryDelay,
  } = {},
) {
  if (typeof token !== 'string' || token.length === 0) {
    throw new Error('A GitHub token is required');
  }
  if (typeof fetchImpl !== 'function') {
    throw new Error('Native fetch is unavailable');
  }

  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    try {
      const response = await fetchImpl(url, {
        method: 'GET',
        headers: {
          Accept: 'application/vnd.github+json',
          Authorization: `Bearer ${token}`,
          'User-Agent': 'paddle-skip-skill-action',
          'X-GitHub-Api-Version': '2022-11-28',
        },
        signal: AbortSignal.timeout(timeoutMs),
      });

      if (response.status === 429 || response.status >= 500) {
        if (attempt === attempts) {
          throw new Error('GitHub API is temporarily unavailable');
        }
        await sleep(attempt);
        continue;
      }
      if (!response.ok) {
        throw new Error(`GitHub API request failed with status ${response.status}`);
      }

      const body = await response.text();
      try {
        return JSON.parse(body);
      } catch {
        throw new Error('GitHub API returned an invalid response');
      }
    } catch (error) {
      if (isNetworkError(error) && attempt < attempts) {
        await sleep(attempt);
        continue;
      }
      throw error;
    }
  }

  throw new Error('GitHub API request failed');
}

function buildApiUrl(apiBase, repository, pullNumber, endpoint, page) {
  const [owner, repo] = repository.split('/');
  const base = apiBase.replace(/\/+$/, '');
  const pullUrl = `${base}/repos/${encodeURIComponent(owner)}/${encodeURIComponent(
    repo,
  )}/pulls/${pullNumber}`;
  if (endpoint === 'files') {
    return `${pullUrl}/files?per_page=${PAGE_SIZE}&page=${page}`;
  }
  return pullUrl;
}

function getEventPullRequest(event) {
  const repository = event?.repository?.full_name;
  const pullNumber = event?.pull_request?.number;
  const headSha = event?.pull_request?.head?.sha;

  if (
    typeof repository !== 'string' ||
    repository.split('/').length !== 2 ||
    repository.split('/').some((part) => part.length === 0) ||
    !Number.isInteger(pullNumber) ||
    pullNumber <= 0 ||
    typeof headSha !== 'string' ||
    !/^[0-9a-f]{40}$/i.test(headSha)
  ) {
    return null;
  }
  return { repository, pullNumber, headSha };
}

function isExpectedPullRequest(
  pullRequest,
  { pullNumber, headSha, changedFiles },
) {
  return (
    pullRequest !== null &&
    typeof pullRequest === 'object' &&
    pullRequest.number === pullNumber &&
    pullRequest.head?.sha === headSha &&
    Number.isInteger(pullRequest.changed_files) &&
    pullRequest.changed_files === changedFiles
  );
}

async function listPullRequestFiles({
  apiBase,
  repository,
  pullNumber,
  expectedCount,
  token,
  request = githubApiRequest,
}) {
  if (
    !Number.isInteger(expectedCount) ||
    expectedCount <= 0 ||
    expectedCount > MAX_CHANGED_FILES
  ) {
    return null;
  }

  const files = [];
  const pages = Math.ceil(expectedCount / PAGE_SIZE);
  for (let page = 1; page <= pages; page += 1) {
    const pageFiles = await request(
      buildApiUrl(apiBase, repository, pullNumber, 'files', page),
      { token },
    );
    if (
      !Array.isArray(pageFiles) ||
      pageFiles.length === 0 ||
      pageFiles.length > PAGE_SIZE
    ) {
      return null;
    }

    files.push(...pageFiles);
    if (files.length > expectedCount) {
      return null;
    }
    if (pageFiles.length < PAGE_SIZE && files.length !== expectedCount) {
      return null;
    }
  }

  return files.length === expectedCount ? files : null;
}

async function shouldSkipPullRequest({
  event,
  token,
  ignorePaths,
  apiBase = DEFAULT_API_URL,
  request = githubApiRequest,
}) {
  const eventPullRequest = getEventPullRequest(event);
  const patterns = parseIgnorePatterns(ignorePaths);
  if (eventPullRequest === null || patterns.length === 0) {
    return false;
  }

  const { repository, pullNumber, headSha } = eventPullRequest;
  const pullUrl = buildApiUrl(apiBase, repository, pullNumber);
  const pullRequest = await request(pullUrl, { token });
  const changedFiles = pullRequest?.changed_files;
  if (
    !isExpectedPullRequest(pullRequest, {
      pullNumber,
      headSha,
      changedFiles,
    }) ||
    changedFiles <= 0 ||
    changedFiles > MAX_CHANGED_FILES
  ) {
    return false;
  }

  const files = await listPullRequestFiles({
    apiBase,
    repository,
    pullNumber,
    expectedCount: changedFiles,
    token,
    request,
  });
  if (files === null) {
    return false;
  }

  // Close the race where the pull request changes while files are paginated.
  const currentPullRequest = await request(pullUrl, { token });
  if (
    !isExpectedPullRequest(currentPullRequest, {
      pullNumber,
      headSha,
      changedFiles,
    })
  ) {
    return false;
  }

  return areAllPathsIgnored(files, patterns);
}

function readEvent(eventPath) {
  if (typeof eventPath !== 'string' || eventPath.length === 0) {
    throw new Error('GITHUB_EVENT_PATH is required');
  }
  return JSON.parse(fs.readFileSync(eventPath, 'utf8'));
}

async function runCheck({ env = process.env, request = githubApiRequest } = {}) {
  const event = readEvent(env.GITHUB_EVENT_PATH);
  if (env.GITHUB_EVENT_NAME !== 'pull_request') {
    return false;
  }

  return shouldSkipPullRequest({
    event,
    token: env['INPUT_GITHUB-TOKEN'],
    ignorePaths: env['INPUT_IGNORE-PATHS'],
    apiBase: env.GITHUB_API_URL || DEFAULT_API_URL,
    request,
  });
}

function writeShouldSkip(outputPath, shouldSkip) {
  fs.appendFileSync(outputPath, `should-skip=${shouldSkip ? 'true' : 'false'}\n`);
}

async function main() {
  let shouldSkip = false;
  try {
    shouldSkip = await runCheck();
  } catch {
    console.warn('::warning::Unable to verify changed files; CI will run.');
  }

  try {
    writeShouldSkip(process.env.GITHUB_OUTPUT, shouldSkip);
  } catch {
    console.warn('::warning::Unable to write the skip-skill action output.');
  }
}

module.exports = {
  areAllPathsIgnored,
  buildApiUrl,
  collectChangedPaths,
  getEventPullRequest,
  githubApiRequest,
  listPullRequestFiles,
  parseIgnorePatterns,
  pathMatchesAnyPattern,
  readEvent,
  runCheck,
  shouldSkipPullRequest,
  writeShouldSkip,
};

if (require.main === module) {
  void main();
}
