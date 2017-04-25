#!/bin/bash
set -e
function usage(){
        echo "usage: build_doc [--help] [<args>]"
        echo "This script generates doc and doc_cn in the script's directory."
        echo "These are common commands used in various situations:"
        echo "    with_docker       build doc and doc_cn with docker"
        echo "    local             build doc and doc_cn locally"
}


case "$1" in
    "with_docker")
        docker run --rm -v $PWD/../../../../:/paddle \
            -e "WITH_GPU=OFF" -e "WITH_AVX=ON" -e "WITH_DOC=ON" paddledev/paddle:dev
        ;;
    "local")
        mkdir -p doc
        mkdir -p doc_cn
        PADDLE_SOURCE_DIR=$PWD/../../../../
        mkdir -p $PADDLE_SOURCE_DIR/build_doc
        pushd $PADDLE_SOURCE_DIR/build_doc
        cmake .. -DWITH_DOC=ON
        make paddle_docs paddle_docs_cn
        popd
        cp -r $PADDLE_SOURCE_DIR/build_doc/doc/en/html/* doc
        cp -r $PADDLE_SOURCE_DIR/build_doc/doc/cn/html/* doc_cn
        rm -rf $PADDLE_SOURCE_DIR/build_doc
        ;;
    "--help")
        usage
        ;;
    *)
        usage
        ;;
esac
