#!/bin/bash
cd `dirname $0`

if [ ${JOB} == "BUILD_AND_TEST" ]; then
  ./build_and_test.sh
elif [ ${JOB} == "DOCS" ]; then
  ./docs.sh
else
  echo Unknown job ${JOB}
  exit 1
fi
