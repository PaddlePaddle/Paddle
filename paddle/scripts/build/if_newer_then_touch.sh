#!/bin/bash
DUMMY_FILE=$1
shift

while [ "$#" != "0" ]
do
    SRC_FILE=$1
    if [ "${DUMMY_FILE}" -ot "${SRC_FILE}" ]; then
        touch ${DUMMY_FILE}
	exit
    fi
    shift
done
