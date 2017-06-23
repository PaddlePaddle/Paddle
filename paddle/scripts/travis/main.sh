#!/bin/bash
cd `dirname $0`

if [ ${JOB} == "DOCS" ]; then
  ./build_doc.sh
elif [ ${JOB} == "PRE_COMMIT" ]; then
  ./check_style.sh
else
  echo "Unknown Travis CI job: ${JOB}"
  exit 0 # Don't fail due to unknown Travis CI job.
fi
