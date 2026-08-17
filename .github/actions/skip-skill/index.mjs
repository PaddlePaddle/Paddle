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

import { appendFileSync, readFileSync } from 'node:fs';
import { matchesGlob } from 'node:path';

const PAGE_SIZE = 100;
const MAX_FILES = 3000;
const sleep = (attempt) =>
  new Promise((resolve) => setTimeout(resolve, attempt * 250));
const isNetworkError = (error) =>
  error instanceof TypeError ||
  error?.name === 'AbortError' ||
  error?.name === 'TimeoutError';

async function githubGet(url, token) {
  if (!token) throw new Error('GitHub token is required');

  for (let attempt = 1; attempt <= 3; attempt += 1) {
    try {
      const response = await fetch(url, {
        headers: {
          Accept: 'application/vnd.github+json',
          Authorization: `Bearer ${token}`,
          'User-Agent': 'paddle-skip-skill-action',
          'X-GitHub-Api-Version': '2022-11-28',
        },
        signal: AbortSignal.timeout(10_000),
      });
      if (response.status === 429 || response.status >= 500) {
        if (attempt === 3) {
          throw new Error('GitHub API is temporarily unavailable');
        }
        await sleep(attempt);
        continue;
      }
      if (!response.ok) {
        throw new Error(`GitHub API request failed (${response.status})`);
      }
      return await response.json();
    } catch (error) {
      if (isNetworkError(error) && attempt < 3) {
        await sleep(attempt);
        continue;
      }
      throw error;
    }
  }
}

function allPathsIgnored(files, patterns) {
  return files.every((file) => {
    if (!file || typeof file.filename !== 'string' || !file.filename) {
      return false;
    }
    const paths = [file.filename];
    if ('previous_filename' in file) {
      if (typeof file.previous_filename !== 'string' || !file.previous_filename) {
        return false;
      }
      paths.push(file.previous_filename);
    } else if (file.status === 'renamed') {
      return false;
    }
    return paths.every((filePath) =>
      patterns.some((pattern) => matchesGlob(filePath, pattern)),
    );
  });
}

async function shouldSkip() {
  const event = JSON.parse(readFileSync(process.env.GITHUB_EVENT_PATH, 'utf8'));
  if (process.env.GITHUB_EVENT_NAME !== 'pull_request') return false;

  const repository = event.repository?.full_name;
  const number = event.pull_request?.number;
  const eventHead = event.pull_request?.head?.sha;
  const patterns = (process.env['INPUT_IGNORE-PATHS'] ?? '')
    .split(',')
    .map((pattern) => pattern.trim())
    .filter(Boolean);
  if (
    typeof repository !== 'string' ||
    repository.split('/').length !== 2 ||
    !Number.isInteger(number) ||
    !/^[0-9a-f]{40}$/i.test(eventHead ?? '') ||
    patterns.length === 0
  ) {
    return false;
  }

  const apiBase = (process.env.GITHUB_API_URL ?? 'https://api.github.com').replace(
    /\/+$/,
    '',
  );
  const [owner, repo] = repository.split('/').map(encodeURIComponent);
  const pullUrl = `${apiBase}/repos/${owner}/${repo}/pulls/${number}`;
  const token = process.env['INPUT_GITHUB-TOKEN'];
  const pull = await githubGet(pullUrl, token);
  const expected = pull.changed_files;
  if (
    pull.number !== number ||
    pull.head?.sha !== eventHead ||
    !Number.isInteger(expected) ||
    expected < 1 ||
    expected > MAX_FILES
  ) {
    return false;
  }

  const files = [];
  const pages = Math.ceil(expected / PAGE_SIZE);
  for (let page = 1; page <= pages; page += 1) {
    const batch = await githubGet(
      `${pullUrl}/files?per_page=${PAGE_SIZE}&page=${page}`,
      token,
    );
    if (
      !Array.isArray(batch) ||
      batch.length < 1 ||
      batch.length > PAGE_SIZE ||
      (page < pages && batch.length !== PAGE_SIZE)
    ) {
      return false;
    }
    files.push(...batch);
  }
  return files.length === expected && allPathsIgnored(files, patterns);
}

let skip = false;
try {
  skip = await shouldSkip();
} catch {
  console.warn('::warning::Unable to verify changed files; CI will run.');
}
try {
  appendFileSync(process.env.GITHUB_OUTPUT, `should-skip=${skip}\n`);
} catch {
  console.warn('::warning::Unable to write the skip-skill action output.');
}
