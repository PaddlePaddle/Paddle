#!/bin/bash
cd `dirname $0`

if [ ${JOB} == "BUILD_AND_TEST" ]; then
  if [ "$TRAVIS_PULL_REQUEST" != "false" ]; then
    TRAVIS_COMMIT_RANGE="FETCH_HEAD...$TRAVIS_BRANCH"
  fi
  git diff --name-only $TRAVIS_COMMIT_RANGE | grep -qvE '(\.md$)' || {
    echo "Only markdown docs were updated, stopping build process."
    exit
  }
  ./build_and_test.sh
elif [ ${JOB} == "DOCS" ]; then
  ./docs.sh
else
  echo Unknown job ${JOB}
  exit 1
fi
