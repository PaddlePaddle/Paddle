#!/bin/bash
cd `dirname $0`

if [ "$TRAVIS_OS_NAME" == "linux" ]; then
  # for manually installed protobuf 3.10
  export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH
fi

if [ ${JOB} == "BUILD_AND_TEST" ]; then
  ./build_and_test.sh
elif [ ${JOB} == "DOCS" ]; then
  ./docs.sh
elif [ ${JOB} == "PRE_COMMIT" ]; then
  ./precommit.sh
else
  echo Unknown job ${JOB}
  exit 1
fi
