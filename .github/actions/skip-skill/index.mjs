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

function isRetryable(error) {
  return (
    error?.retryable === true ||
    error instanceof TypeError ||
    error?.name === 'AbortError' ||
    error?.name === 'TimeoutError'
  );
}

function retry(fn, shouldRetry, attempts) {
  return async (...args) => {
    for (let attempt = 1; attempt <= attempts; attempt += 1) {
      try {
        return await fn(...args);
      } catch (error) {
        if (!shouldRetry(error) || attempt === attempts) throw error;
        await sleep(attempt);
      }
    }
  };
}

async function getJson(url, token) {
  if (!token) throw new Error('GitHub token is required');
  const response = await fetch(url, {
    headers: {
      Accept: 'application/vnd.github+json',
      Authorization: `Bearer ${token}`,
      'User-Agent': 'paddle-skip-skill-action',
      'X-GitHub-Api-Version': '2022-11-28',
    },
    signal: AbortSignal.timeout(10_000),
  });
  if (!response.ok) {
    throw Object.assign(
      new Error(`GitHub API request failed (${response.status})`),
      { retryable: response.status === 429 || response.status >= 500 },
    );
  }
  return response.json();
}

const githubGet = retry(getJson, isRetryable, 3);

function withAllPages(getPage) {
  return async (url, token, expected) => {
    const files = [];
    const pages = Math.ceil(expected / PAGE_SIZE);
    for (let page = 1; page <= pages; page += 1) {
      const batch = await getPage(
        `${url}?per_page=${PAGE_SIZE}&page=${page}`,
        token,
      );
      const expectedSize = Math.min(PAGE_SIZE, expected - files.length);
      if (!Array.isArray(batch) || batch.length !== expectedSize) return null;
      files.push(...batch);
    }
    return files;
  };
}

const listFiles = withAllPages(githubGet);

function filePaths(file) {
  if (!file || typeof file.filename !== 'string' || !file.filename) return null;
  switch (file.status) {
    case 'renamed':
      return typeof file.previous_filename === 'string' && file.previous_filename
        ? [file.filename, file.previous_filename]
        : null;
    case 'added':
    case 'removed':
    case 'modified':
    case 'copied':
    case 'changed':
    case 'unchanged':
      return [file.filename];
    default:
      return null;
  }
}

function allPathsIgnored(files, patterns) {
  return files.every((file) => {
    const paths = filePaths(file);
    return (
      paths !== null &&
      paths.every((filePath) =>
        patterns.some((pattern) => matchesGlob(filePath, pattern)),
      )
    );
  });
}

function pullContext(event) {
  const repository = event.repository?.full_name;
  const parts = typeof repository === 'string' ? repository.split('/') : [];
  const number = event.pull_request?.number;
  const head = event.pull_request?.head?.sha;
  if (
    parts.length !== 2 ||
    parts.some((part) => !part) ||
    !Number.isInteger(number) ||
    number < 1 ||
    !/^[0-9a-f]{40}$/i.test(head ?? '')
  ) {
    return null;
  }
  return { repository: parts.map(encodeURIComponent).join('/'), number, head };
}

function parsePatterns(value) {
  return (value ?? '')
    .split(',')
    .map((pattern) => pattern.trim())
    .filter(Boolean);
}

async function shouldSkip() {
  const event = JSON.parse(readFileSync(process.env.GITHUB_EVENT_PATH, 'utf8'));
  if (process.env.GITHUB_EVENT_NAME !== 'pull_request') return false;

  const context = pullContext(event);
  const patterns = parsePatterns(process.env['INPUT_IGNORE-PATHS']);
  if (context === null || patterns.length === 0) return false;

  const api = (process.env.GITHUB_API_URL ?? 'https://api.github.com').replace(
    /\/+$/,
    '',
  );
  const pullUrl = `${api}/repos/${context.repository}/pulls/${context.number}`;
  const token = process.env['INPUT_GITHUB-TOKEN'];
  const pull = await githubGet(pullUrl, token);
  const expected = pull.changed_files;
  if (
    pull.number !== context.number ||
    pull.head?.sha !== context.head ||
    !Number.isInteger(expected) ||
    expected < 1 ||
    expected > MAX_FILES
  ) {
    return false;
  }

  const files = await listFiles(`${pullUrl}/files`, token, expected);
  return files !== null && allPathsIgnored(files, patterns);
}

async function run() {
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
}

await run();
