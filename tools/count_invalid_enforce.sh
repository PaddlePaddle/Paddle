#!/bin/bash
ALL_PADDLE_CHECK=`grep -r -zoE "(PADDLE_ENFORCE[A-Z_]*|PADDLE_THROW)\(.[^,\);]*.[^;]*\);\s" ../paddle/fluid || true`
ALL_PADDLE_CHECK_CNT=`echo "$ALL_PADDLE_CHECK" | grep -cE "(PADDLE_ENFORCE|PADDLE_THROW)" || true`
VALID_PADDLE_CHECK_CNT=`echo "$ALL_PADDLE_CHECK" | grep -zoE '(PADDLE_ENFORCE[A-Z_]*|PADDLE_THROW)\((.[^,;]+,)*.[^";]*(errors::).[^"]*".[^";]{20,}.[^;]*\);\s' | grep -cE "(PADDLE_ENFORCE|PADDLE_THROW)" || true`

echo "----------------------------"
echo "PADDLE ENFORCE & THROW COUNT"
echo "----------------------------"
echo "All PADDLE_ENFORCE{_**} & PADDLE_THROW Count: ${ALL_PADDLE_CHECK_CNT}"
echo "Valid PADDLE_ENFORCE{_**} & PADDLE_THROW Count: ${VALID_PADDLE_CHECK_CNT}"
echo "Invalid PADDLE_ENFORCE{_**} & PADDLE_THROW Count: $(($ALL_PADDLE_CHECK_CNT-$VALID_PADDLE_CHECK_CNT))"
