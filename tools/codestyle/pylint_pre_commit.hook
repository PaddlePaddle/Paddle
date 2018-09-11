#!/bin/bash

TOTAL_ERRORS=0


DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$DIR:$PYTHONPATH

# The trick to remove deleted files: https://stackoverflow.com/a/2413151
for file in $(git diff --name-status | awk '$1 != "D" {print $2}'); do
    pylint --disable=all --load-plugins=docstring_checker \
    --enable=doc-string-one-line,doc-string-end-with,doc-string-with-all-args,doc-string-triple-quotes,doc-string-missing,doc-string-indent-error,doc-string-with-returns,doc-string-with-raises $file;
    TOTAL_ERRORS=$(expr $TOTAL_ERRORS + $?);
done

exit $TOTAL_ERRORS
#For now, just warning:
#exit 0

