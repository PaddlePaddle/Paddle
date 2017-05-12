#!/bin/bash
docker run --rm \
       -v $(git rev-parse --show-toplevel):/paddle \
       -e "WITH_GPU=OFF" \
       -e "WITH_AVX=ON" \
       -e "WITH_DOC=ON" \
       -e "WOBOQ=ON" \
       ${1:-"paddledev/paddle:dev"}
