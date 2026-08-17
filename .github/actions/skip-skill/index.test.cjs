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

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const { test } = require('node:test');

const {
  areAllPathsIgnored,
  githubApiRequest,
  listPullRequestFiles,
  parseIgnorePatterns,
  pathMatchesAnyPattern,
  runCheck,
  shouldSkipPullRequest,
} = require('./index.cjs');

const HEAD_SHA = 'a'.repeat(40);
const REPOSITORY = 'PaddlePaddle/Paddle';
const PULL_NUMBER = 123;
const TOKEN = 'test-token';

function eventFixture() {
  return {
    repository: { full_name: REPOSITORY },
    pull_request: {
      number: PULL_NUMBER,
      head: { sha: HEAD_SHA },
    },
  };
}

function pullFixture(changedFiles, headSha = HEAD_SHA) {
  return {
    number: PULL_NUMBER,
    head: { sha: headSha },
    changed_files: changedFiles,
  };
}

function response(status, body) {
  return {
    status,
    ok: status >= 200 && status < 300,
    text: async () => JSON.stringify(body),
  };
}

function writeEvent(event) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'skip-skill-test-'));
  const eventPath = path.join(directory, 'event.json');
  fs.writeFileSync(eventPath, JSON.stringify(event));
  return { directory, eventPath };
}

test('parses trimmed comma-separated ignore patterns', () => {
  assert.deepEqual(
    parseIgnorePatterns(' .agents/skills/**, docs/** ,,**.md '),
    ['.agents/skills/**', 'docs/**', '**.md'],
  );
  assert.deepEqual(parseIgnorePatterns(undefined), []);
});

test('matches repository paths with native glob semantics', () => {
  const patterns = ['.agents/skills/**', 'docs/**'];
  assert.equal(
    pathMatchesAnyPattern('.agents/skills/paddle-build/SKILL.md', patterns),
    true,
  );
  assert.equal(pathMatchesAnyPattern('paddle/fluid/framework.cc', patterns), false);
});

test('requires both current and previous filenames to be ignored', () => {
  const patterns = ['.agents/skills/**'];
  assert.equal(
    areAllPathsIgnored(
      [
        { filename: '.agents/skills/new.md', status: 'modified' },
        {
          filename: '.agents/skills/new-name.md',
          previous_filename: '.agents/skills/old-name.md',
          status: 'renamed',
        },
      ],
      patterns,
    ),
    true,
  );
  assert.equal(
    areAllPathsIgnored(
      [
        {
          filename: '.agents/skills/new-name.md',
          previous_filename: 'python/old-name.py',
          status: 'renamed',
        },
      ],
      patterns,
    ),
    false,
  );
  assert.equal(
    areAllPathsIgnored(
      [{ filename: '.agents/skills/new-name.md', status: 'renamed' }],
      patterns,
    ),
    false,
  );
});

test('returns false for non-pull-request events without calling the API', async (t) => {
  const { directory, eventPath } = writeEvent({ repository: {} });
  t.after(() => fs.rmSync(directory, { recursive: true }));
  let apiCalls = 0;

  const result = await runCheck({
    env: {
      GITHUB_EVENT_NAME: 'push',
      GITHUB_EVENT_PATH: eventPath,
    },
    request: async () => {
      apiCalls += 1;
      throw new Error('unexpected API call');
    },
  });

  assert.equal(result, false);
  assert.equal(apiCalls, 0);
});

test('skips when every complete API file entry matches an ignore pattern', async () => {
  const calls = [];
  const results = [
    pullFixture(2),
    [
      { filename: '.agents/skills/first/SKILL.md', status: 'modified' },
      {
        filename: '.agents/skills/new.md',
        previous_filename: '.agents/skills/old.md',
        status: 'renamed',
      },
    ],
    pullFixture(2),
  ];
  const request = async (url, options) => {
    calls.push({ url, options });
    return results.shift();
  };

  const result = await shouldSkipPullRequest({
    event: eventFixture(),
    token: TOKEN,
    ignorePaths: ' .agents/skills/** ',
    request,
  });

  assert.equal(result, true);
  assert.equal(calls.length, 3);
  assert.equal(
    calls[0].url,
    `https://api.github.com/repos/PaddlePaddle/Paddle/pulls/${PULL_NUMBER}`,
  );
  assert.equal(
    calls[1].url,
    `https://api.github.com/repos/PaddlePaddle/Paddle/pulls/${PULL_NUMBER}/files?per_page=100&page=1`,
  );
  assert.deepEqual(calls.map((call) => call.options), [
    { token: TOKEN },
    { token: TOKEN },
    { token: TOKEN },
  ]);
});

test('does not skip when the API head SHA differs from the event', async () => {
  let apiCalls = 0;
  const result = await shouldSkipPullRequest({
    event: eventFixture(),
    token: TOKEN,
    ignorePaths: '.agents/skills/**',
    request: async () => {
      apiCalls += 1;
      return pullFixture(1, 'b'.repeat(40));
    },
  });

  assert.equal(result, false);
  assert.equal(apiCalls, 1);
});

test('does not request file pages when changed_files exceeds the API limit', async () => {
  let apiCalls = 0;
  const result = await shouldSkipPullRequest({
    event: eventFixture(),
    token: TOKEN,
    ignorePaths: '.agents/skills/**',
    request: async () => {
      apiCalls += 1;
      return pullFixture(3001);
    },
  });

  assert.equal(result, false);
  assert.equal(apiCalls, 1);
});

test('rejects empty and incomplete file pages', async () => {
  for (const pageResult of [[], [{ filename: '.agents/skills/only.md' }]]) {
    let apiCalls = 0;
    const result = await shouldSkipPullRequest({
      event: eventFixture(),
      token: TOKEN,
      ignorePaths: '.agents/skills/**',
      request: async () => {
        apiCalls += 1;
        return apiCalls === 1 ? pullFixture(2) : pageResult;
      },
    });

    assert.equal(result, false);
    assert.equal(apiCalls, 2);
  }
});

test('paginates 100 files per page and requires the exact expected count', async () => {
  const files = Array.from({ length: 101 }, (_, index) => ({
    filename: `.agents/skills/file-${index}.md`,
    status: 'modified',
  }));
  const seenUrls = [];
  const request = async (url) => {
    seenUrls.push(url);
    if (!url.includes('/files?')) {
      return pullFixture(files.length);
    }
    return url.endsWith('page=1') ? files.slice(0, 100) : files.slice(100);
  };

  const result = await shouldSkipPullRequest({
    event: eventFixture(),
    token: TOKEN,
    ignorePaths: '.agents/skills/**',
    request,
  });

  assert.equal(result, true);
  assert.equal(seenUrls.filter((url) => url.includes('/files?')).length, 2);
  assert.equal(seenUrls.some((url) => url.endsWith('page=2')), true);
});

test('rejects a pull request that changes while file pages are read', async () => {
  const results = [
    pullFixture(1),
    [{ filename: '.agents/skills/only.md', status: 'modified' }],
    pullFixture(2),
  ];

  const result = await shouldSkipPullRequest({
    event: eventFixture(),
    token: TOKEN,
    ignorePaths: '.agents/skills/**',
    request: async () => results.shift(),
  });

  assert.equal(result, false);
});

test('listPullRequestFiles rejects overfilled pages', async () => {
  const result = await listPullRequestFiles({
    apiBase: 'https://api.github.com',
    repository: REPOSITORY,
    pullNumber: PULL_NUMBER,
    expectedCount: 1,
    token: TOKEN,
    request: async () => [
      { filename: '.agents/skills/one.md' },
      { filename: '.agents/skills/two.md' },
    ],
  });

  assert.equal(result, null);
});

test('GitHub API requests use the expected headers and retry network errors', async () => {
  const calls = [];
  const sleeps = [];
  const result = await githubApiRequest('https://api.github.test/resource', {
    token: TOKEN,
    fetchImpl: async (url, options) => {
      calls.push({ url, options });
      if (calls.length === 1) {
        throw new TypeError('network unavailable');
      }
      return response(200, { ok: true });
    },
    sleep: async (attempt) => sleeps.push(attempt),
  });

  assert.deepEqual(result, { ok: true });
  assert.equal(calls.length, 2);
  assert.deepEqual(sleeps, [1]);
  assert.equal(calls[0].options.headers.Accept, 'application/vnd.github+json');
  assert.equal(calls[0].options.headers.Authorization, `Bearer ${TOKEN}`);
  assert.equal(calls[0].options.signal instanceof AbortSignal, true);
});

test('GitHub API requests retry 429 and 5xx responses', async () => {
  const statuses = [429, 503, 200];
  const sleeps = [];

  const result = await githubApiRequest('https://api.github.test/resource', {
    token: TOKEN,
    fetchImpl: async () => response(statuses.shift(), { ok: true }),
    sleep: async (attempt) => sleeps.push(attempt),
  });

  assert.deepEqual(result, { ok: true });
  assert.deepEqual(sleeps, [1, 2]);
});

test('GitHub API requests do not retry non-retryable HTTP or JSON errors', async () => {
  for (const fetchImpl of [
    async () => response(404, { message: 'not found' }),
    async () => ({ status: 200, ok: true, text: async () => 'not-json' }),
  ]) {
    let calls = 0;
    await assert.rejects(
      githubApiRequest('https://api.github.test/resource', {
        token: TOKEN,
        fetchImpl: async (...args) => {
          calls += 1;
          return fetchImpl(...args);
        },
        sleep: async () => {},
      }),
    );
    assert.equal(calls, 1);
  }
});

test('the action writes false and exits normally when verification fails', (t) => {
  const { directory, eventPath } = writeEvent(eventFixture());
  const outputPath = path.join(directory, 'output.txt');
  t.after(() => fs.rmSync(directory, { recursive: true }));

  const result = spawnSync(process.execPath, [path.join(__dirname, 'index.cjs')], {
    encoding: 'utf8',
    env: {
      ...process.env,
      GITHUB_EVENT_NAME: 'pull_request',
      GITHUB_EVENT_PATH: eventPath,
      GITHUB_OUTPUT: outputPath,
      'INPUT_IGNORE-PATHS': '.agents/skills/**',
      'INPUT_GITHUB-TOKEN': '',
    },
  });

  assert.equal(result.status, 0);
  assert.equal(fs.readFileSync(outputPath, 'utf8'), 'should-skip=false\n');
  assert.match(result.stderr, /CI will run/);
});
