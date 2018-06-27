#!/bin/bash

TOTAL_ERRORS=0

# The trick to remove deleted files: https://stackoverflow.com/a/2413151
for file in $(git diff --cached --name-status | awk '$1 != "D" {print $2}'); do
    if [[ $file =~ ^(paddle/api/.*|paddle/capi/.*|paddle/contrib/.*|paddle/cuda/.*|paddle/function/.*|paddle/gserver/.*|paddle/math/.*|paddle/optimizer/.*|paddle/parameter/.*|paddle/pserver/.*|paddle/trainer/.*|paddle/utils/.*) ]]; then
        continue;
    else
        cpplint --filter=-readability/fn_size $file;
        TOTAL_ERRORS=$(expr $TOTAL_ERRORS + $?);
    fi
done

exit $TOTAL_ERRORS

