# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING
from urllib import error, request

if TYPE_CHECKING:
    from collections.abc import Iterable

DEFAULT_OWNER = "PaddlePaddle"
DEFAULT_REPO = "Paddle"
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
PR_NUMBER_RE = re.compile(r"\(#(\d+)\)")
HEADING_RE = re.compile(r"(?m)^###\s*(.+?)\s*$")
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

PR_CATEGORIES = [
    'User Experience',
    'Execute Infrastructure',
    'Operator Mechanism',
    'CINN',
    'Custom Device',
    'Performance Optimization',
    'Distributed Strategy',
    'Parameter Server',
    'Communication Library',
    'Auto Parallel',
    'Inference',
    'Environment Adaptation',
    'Others',
]

PR_TYPES = [
    'New features',
    'Bug fixes',
    'Improvements',
    'Performance',
    'BC Breaking',
    'Deprecations',
    'Docs',
    'Devs',
    'Not User Facing',
    'Security',
    'Others',
]


@dataclass(frozen=True)
class CommitRecord:
    commit_hash: str
    title: str
    git_author: str
    pr_number: int | None


@dataclass(frozen=True)
class PullRequestRecord:
    number: int
    title: str
    author: str
    labels: list[str]
    reviewers: list[str]
    category: str
    topic: str
    description: str


@dataclass(frozen=True)
class CommitRow:
    commit_hash: str
    category: str
    topic: str
    title: str
    pr_link: str
    author: str
    labels: str
    accepter_1: str
    accepter_2: str
    accepter_3: str
    description: str


COMMIT_FIELDS = tuple(field.name for field in fields(CommitRow))


def run_git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip()


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def strip_html_comments(text: str) -> str:
    return HTML_COMMENT_RE.sub("", text or "")


def cleanup_markdown_text(text: str) -> str:
    cleaned_lines = []
    for line in strip_html_comments(text).replace("\r\n", "\n").splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^[-*+]\s*", "", line)
        line = re.sub(r"^\d+\.\s*", "", line)
        cleaned_lines.append(line)
    return normalize_whitespace(" ".join(cleaned_lines))


def cleanup_title(title: str) -> str:
    return normalize_whitespace(PR_NUMBER_RE.sub("", title or ""))


def extract_pr_number(title: str) -> int | None:
    matches = PR_NUMBER_RE.findall(title or "")
    return int(matches[-1]) if matches else None


def split_known_choice(raw_text: str, choices: list[str], default: str) -> str:
    parts = [
        cleanup_markdown_text(part)
        for part in re.split(r'[,|\n]+', raw_text or '')
    ]
    parts = [part for part in parts if part]
    if not parts:
        return default
    for part in parts:
        for choice in choices:
            if part.casefold() == choice.casefold():
                return choice
    for part in parts:
        lowered = part.casefold()
        for choice in choices:
            if choice.casefold() in lowered:
                return choice
    return default


def label_value(labels: Iterable[str], prefix: str) -> str | None:
    prefix_lower = prefix.casefold()
    for label in labels:
        if label.casefold().startswith(prefix_lower):
            _, value = label.split(':', 1)
            return value.strip()
    return None


def markdown_sections(body: str) -> dict[str, str]:
    cleaned = (
        strip_html_comments(body).replace("\r\n", "\n").replace("\r", "\n")
    )
    matches = list(HEADING_RE.finditer(cleaned))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        start = match.end()
        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(cleaned)
        )
        sections[normalize_whitespace(match.group(1))] = cleaned[
            start:end
        ].strip()
    return sections


def section_value(sections: dict[str, str], name: str) -> str:
    for key, value in sections.items():
        if key.casefold() == name.casefold():
            return value
    return ""


def resolve_category(labels: list[str], body: str) -> str:
    value = label_value(labels, 'release notes:')
    if value:
        return split_known_choice(value, PR_CATEGORIES, 'Others')
    sections = markdown_sections(body)
    return split_known_choice(
        section_value(sections, 'PR Category'), PR_CATEGORIES, 'Others'
    )


def resolve_topic(labels: list[str], body: str) -> str:
    value = label_value(labels, 'topic:')
    if value:
        return split_known_choice(value, PR_TYPES, 'Others')
    sections = markdown_sections(body)
    return split_known_choice(
        section_value(sections, 'PR Types'), PR_TYPES, 'Others'
    )


def resolve_description(body: str) -> str:
    sections = markdown_sections(body)
    return cleanup_markdown_text(section_value(sections, 'Description'))


def load_token(explicit_token: str | None) -> str | None:
    if explicit_token:
        return explicit_token
    for env_name in ('GITHUB_API_TOKEN', 'GITHUB_TOKEN', 'GH_TOKEN'):
        value = os.getenv(env_name)
        if value:
            return value
    token_file = Path('~/.gh_tokenrc').expanduser()
    if token_file.exists():
        match = re.search(r'github_oauth\s*=\s*(\S+)', token_file.read_text())
        if match:
            return match.group(1)
    return None


def collect_commits(
    base_ref: str, head_ref: str, use_merge_base: bool
) -> list[CommitRecord]:
    start_ref = base_ref
    if use_merge_base:
        start_ref = run_git('merge-base', base_ref, head_ref)
    raw_log = run_git(
        'log',
        '--reverse',
        '--pretty=format:%H%x1f%s%x1f%an%x1e',
        f'{start_ref}..{head_ref}',
    )
    if not raw_log:
        return []

    commits = []
    for record in raw_log.split('\x1e'):
        record = record.strip()
        if not record:
            continue
        commit_hash, title, author = record.split('\x1f')
        commits.append(
            CommitRecord(
                commit_hash=commit_hash,
                title=title,
                git_author=author,
                pr_number=extract_pr_number(title),
            )
        )
    return commits


class PullRequestCache:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, dict[str, object]] = {}
        if self.path.exists():
            self._data = json.loads(self.path.read_text())

    def get_many(self, numbers: Iterable[int]) -> dict[int, PullRequestRecord]:
        records = {}
        for number in numbers:
            cached = self._data.get(str(number))
            if cached is not None:
                cached = dict(cached)
                cached.setdefault('title', f'PR #{number}')
                records[number] = PullRequestRecord(**cached)
        return records

    def update_many(self, records: Iterable[PullRequestRecord]) -> None:
        changed = False
        for record in records:
            self._data[str(record.number)] = asdict(record)
            changed = True
        if changed:
            self.path.write_text(
                json.dumps(self._data, indent=2, sort_keys=True) + '\n'
            )


class GitHubClient:
    def __init__(self, token: str | None, owner: str, repo: str):
        self.owner = owner
        self.repo = repo
        self.headers = {'Accept': 'application/vnd.github+json'}
        if token:
            self.headers['Authorization'] = f'bearer {token}'

    def _request_graphql(self, query: str) -> dict[str, object]:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                payload = json.dumps({'query': query}).encode()
                http_request = request.Request(
                    'https://api.github.com/graphql',
                    data=payload,
                    headers={
                        **self.headers,
                        'Content-Type': 'application/json',
                    },
                    method='POST',
                )
                with request.urlopen(http_request, timeout=60) as response:
                    payload = json.loads(response.read().decode())
                if payload.get('errors'):
                    raise RuntimeError(
                        f"GitHub GraphQL query failed: {payload['errors']}"
                    )
                return payload
            except error.HTTPError as http_error:
                last_error = http_error
                if http_error.code in RETRYABLE_STATUS_CODES and attempt < 2:
                    time.sleep(1 + attempt)
                    continue
                raise
            except (error.URLError, ValueError, RuntimeError) as request_error:
                last_error = request_error
                if attempt == 2:
                    raise
                time.sleep(1 + attempt)
        assert last_error is not None
        raise last_error

    def _fetch_batch(self, batch: list[int]) -> dict[int, PullRequestRecord]:
        query_parts = []
        for number in batch:
            query_parts.append(
                f'''
                pr_{number}: pullRequest(number: {number}) {{
                  title
                  author {{ login }}
                  body
                  labels(first: 50) {{
                    nodes {{ name }}
                  }}
                  reviews(first: 100, states: APPROVED) {{
                    nodes {{
                      author {{ login }}
                    }}
                  }}
                }}
                '''
            )
        payload = self._request_graphql(
            f'''
            query {{
              repository(owner: "{self.owner}", name: "{self.repo}") {{
                {''.join(query_parts)}
              }}
            }}
            '''
        )
        repository = payload['data']['repository']
        results: dict[int, PullRequestRecord] = {}
        for number in batch:
            node = repository.get(f'pr_{number}')
            if node is None:
                continue
            labels = [item['name'] for item in node['labels']['nodes']]
            reviewers = sorted(
                {
                    review['author']['login']
                    for review in node['reviews']['nodes']
                    if review.get('author')
                }
            )
            body = node.get('body') or ''
            results[number] = PullRequestRecord(
                number=number,
                title=node.get('title') or f'PR #{number}',
                author=(node.get('author') or {}).get('login', ''),
                labels=labels,
                reviewers=reviewers,
                category=resolve_category(labels, body),
                topic=resolve_topic(labels, body),
                description=resolve_description(body),
            )
        return results

    def fetch_pull_requests(
        self, numbers: list[int], batch_size: int, workers: int
    ) -> dict[int, PullRequestRecord]:
        unique_numbers = sorted(set(numbers))
        if not unique_numbers:
            return {}
        batches = [
            unique_numbers[index : index + batch_size]
            for index in range(0, len(unique_numbers), batch_size)
        ]
        results: dict[int, PullRequestRecord] = {}
        if workers <= 1 or len(batches) == 1:
            for batch in batches:
                results.update(self._fetch_batch(batch))
            return results

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_batch = {
                executor.submit(self._fetch_batch, batch): batch
                for batch in batches
            }
            for future in as_completed(future_to_batch):
                results.update(future.result())
        return results


def build_commit_rows(
    commits: list[CommitRecord],
    pull_requests: dict[int, PullRequestRecord],
    owner: str,
    repo: str,
) -> tuple[list[CommitRow], list[int]]:
    rows: list[CommitRow] = []
    missing_prs: set[int] = set()

    for commit in commits:
        if commit.pr_number is None:
            rows.append(
                CommitRow(
                    commit_hash=commit.commit_hash,
                    category='Others',
                    topic='Others',
                    title=commit.title,
                    pr_link='',
                    author=commit.git_author,
                    labels='',
                    accepter_1='',
                    accepter_2='',
                    accepter_3='',
                    description='',
                )
            )
            continue

        pr_record = pull_requests.get(commit.pr_number)
        if pr_record is None:
            missing_prs.add(commit.pr_number)
            rows.append(
                CommitRow(
                    commit_hash=commit.commit_hash,
                    category='Others',
                    topic='Others',
                    title=commit.title,
                    pr_link=f'https://github.com/{owner}/{repo}/pull/{commit.pr_number}',
                    author=commit.git_author,
                    labels='',
                    accepter_1='',
                    accepter_2='',
                    accepter_3='',
                    description='',
                )
            )
            continue

        accepters = [*pr_record.reviewers, '', '', ''][:3]
        rows.append(
            CommitRow(
                commit_hash=commit.commit_hash,
                category=pr_record.category,
                topic=pr_record.topic,
                title=pr_record.title or commit.title,
                pr_link=f'https://github.com/{owner}/{repo}/pull/{commit.pr_number}',
                author=pr_record.author or commit.git_author,
                labels=','.join(pr_record.labels),
                accepter_1=accepters[0],
                accepter_2=accepters[1],
                accepter_3=accepters[2],
                description=pr_record.description,
            )
        )

    return rows, sorted(missing_prs)


def ordered_values(
    items: Iterable[str], preferred_order: list[str]
) -> list[str]:
    values = {item for item in items if item}
    ordered = [value for value in preferred_order if value in values]
    extras = sorted(values - set(preferred_order), key=str.casefold)
    return ordered + extras


def get_hash_or_pr_url(commit: CommitRow) -> str:
    if not commit.pr_link:
        return commit.commit_hash
    matches = re.findall(
        r'https://github.com/[^/]+/[^/]+/pull/(\d+)',
        commit.pr_link,
    )
    if not matches:
        return commit.commit_hash
    return f'[#{matches[0]}]({commit.pr_link})'


def markdown_entry_text(commit: CommitRow) -> str:
    return cleanup_title(commit.title)


def category_output_name(category: str) -> str:
    cleaned = re.sub(r'[^0-9A-Za-z._-]+', '_', category.strip())
    cleaned = cleaned.strip('._')
    return cleaned or 'Others'


def get_markdown_header(category: str) -> str:
    return (
        f'# Release Notes worksheet {category}\n'
        '- polish PR title to make it human read friendly.\n'
        '- edit, delete, merge multiple PRs.\n'
        '- summarize notes for this category.\n\n'
    )


def render_markdown(
    commits: list[CommitRow], base_ref: str, head_ref: str
) -> str:
    lines = [
        '# Release Notes\n\n',
        f'- Range: `{base_ref}..{head_ref}`\n',
        f'- Commits: {len(commits)}\n\n',
    ]

    categories = ordered_values(
        (commit.category for commit in commits), PR_CATEGORIES
    )
    for category in categories:
        lines.append(f'## {category}\n\n')
        category_commits = [
            commit for commit in commits if commit.category == category
        ]
        topics = ordered_values(
            (commit.topic for commit in category_commits), PR_TYPES
        )
        for topic in topics:
            topic_commits = [
                commit for commit in category_commits if commit.topic == topic
            ]
            if not topic_commits:
                continue
            lines.append(f'### {topic}\n\n')
            for commit in topic_commits:
                lines.append(
                    f'- {markdown_entry_text(commit)} ({get_hash_or_pr_url(commit)})\n'
                )
            lines.append('\n')
    return ''.join(lines)


def render_category_markdown(category: str, commits: list[CommitRow]) -> str:
    lines = [get_markdown_header(category), f'## {category}\n\n']
    topics = ordered_values((commit.topic for commit in commits), PR_TYPES)
    for topic in topics:
        topic_commits = [commit for commit in commits if commit.topic == topic]
        if not topic_commits:
            continue
        lines.append(f'### {topic}\n\n')
        for commit in topic_commits:
            lines.append(
                f'- {markdown_entry_text(commit)} ({get_hash_or_pr_url(commit)})\n'
            )
        lines.append('\n')
    return ''.join(lines)


def write_csv(path: Path, rows: list[CommitRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(COMMIT_FIELDS)
        for row in rows:
            writer.writerow([getattr(row, field) for field in COMMIT_FIELDS])


def write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(f'{line}\n' for line in lines))


def write_category_exports(
    export_dir: Path, rows: list[CommitRow]
) -> list[Path]:
    export_dir.mkdir(parents=True, exist_ok=True)
    written_files: list[Path] = []
    categories = ordered_values((row.category for row in rows), PR_CATEGORIES)
    for category in categories:
        category_rows = [row for row in rows if row.category == category]
        if not category_rows:
            continue
        category_name = category_output_name(category)
        category_dir = export_dir / category_name
        markdown_path = category_dir / f'result_{category_name}.md'
        csv_path = category_dir / f'result_{category_name}.csv'
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(
            render_category_markdown(category, category_rows)
        )
        write_csv(csv_path, category_rows)
        written_files.extend([markdown_path, csv_path])
    return written_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate release-note CSV and Markdown between two commits in one command.'
    )
    parser.add_argument('base_ref', help='Base git ref or commit')
    parser.add_argument('head_ref', help='Head git ref or commit')
    parser.add_argument(
        '--output-dir',
        default='results',
        help='Directory for generated files (default: results)',
    )
    parser.add_argument(
        '--owner',
        default=DEFAULT_OWNER,
        help=f'GitHub owner (default: {DEFAULT_OWNER})',
    )
    parser.add_argument(
        '--repo',
        default=DEFAULT_REPO,
        help=f'GitHub repo (default: {DEFAULT_REPO})',
    )
    parser.add_argument(
        '--token',
        help='GitHub token; falls back to env vars or ~/.gh_tokenrc',
    )
    parser.add_argument(
        '--cache-path',
        default='results/pr_cache.json',
        help='PR metadata cache path (default: results/pr_cache.json)',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=25,
        help='How many PRs to fetch per GitHub GraphQL request (default: 25)',
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='How many GraphQL batches to fetch concurrently (default: 4)',
    )
    parser.add_argument(
        '--direct-range',
        action='store_true',
        help='Use base_ref..head_ref directly instead of merge-base(base_ref, head_ref)..head_ref',
    )
    parser.add_argument(
        '--local-only',
        action='store_true',
        help='Skip GitHub API calls and only use local git metadata',
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    csv_path = output_dir / 'commitlist.csv'
    export_dir = output_dir / 'export'
    contributors_path = output_dir / 'contributors.txt'

    commits = collect_commits(
        args.base_ref,
        args.head_ref,
        use_merge_base=not args.direct_range,
    )
    if not commits:
        raise SystemExit(
            f'No commits found between {args.base_ref} and {args.head_ref}.'
        )

    pull_requests: dict[int, PullRequestRecord] = {}
    pr_numbers = sorted(
        {commit.pr_number for commit in commits if commit.pr_number is not None}
    )
    if args.local_only and pr_numbers:
        print(
            'Warning: running in --local-only mode. PR metadata will not be fetched, '
            'so category/topic fall back to Others and descriptions stay empty.',
            file=sys.stderr,
        )
    if not args.local_only and pr_numbers:
        token = load_token(args.token)
        if not token:
            print(
                'Warning: GitHub token not found. Falling back to unauthenticated '
                'GitHub API requests; this may hit rate limits on large ranges.',
                file=sys.stderr,
            )
        cache = PullRequestCache(Path(args.cache_path))
        cached = cache.get_many(pr_numbers)
        missing = [
            number
            for number in pr_numbers
            if number not in cached or cached[number].title == f'PR #{number}'
        ]
        fetched = GitHubClient(
            args.token or token, args.owner, args.repo
        ).fetch_pull_requests(
            missing,
            batch_size=max(1, args.batch_size),
            workers=max(1, args.workers),
        )
        cache.update_many(fetched.values())
        pull_requests = {**cached, **fetched}

    rows, missing_prs = build_commit_rows(
        commits, pull_requests, args.owner, args.repo
    )
    contributors = sorted(
        {commit.git_author for commit in commits if commit.git_author},
        key=str.casefold,
    )

    write_csv(csv_path, rows)
    exported_files = write_category_exports(export_dir, rows)
    write_lines(contributors_path, contributors)

    if missing_prs and not args.local_only:
        print(
            'Warning: failed to fetch metadata for PRs '
            + ', '.join(f'#{number}' for number in missing_prs)
            + '. Fallback commit metadata was used.',
            file=sys.stderr,
        )

    print(f'Generated {len(rows)} rows from {len(commits)} commits.')
    print(f'CSV: {csv_path}')
    print(f'Export dir: {export_dir}')
    print(f'Category exports: {len(exported_files)} files')
    print(f'Contributors: {contributors_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
