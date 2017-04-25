#!/bin/bash
set -e
function usage(){
        echo "usage: build_doc [--help] [<args>]"
        echo "This script generates doc and doc_cn in the script's directory."
        echo "These are common commands used in various situations:"
        echo "    with_docker       build doc and doc_cn with docker"
        echo "    local             build doc and doc_cn locally"
}


MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PADDLE_SOURCE_DIR=$MYDIR/../../../../
case "$1" in
    "with_docker")
        docker run --rm -v $PADDLE_SOURCE_DIR:/paddle \
            -e "WITH_GPU=OFF" -e "WITH_AVX=ON" -e "WITH_DOC=ON" paddledev/paddle:dev
        ;;
    "local")
        mkdir -p $MYDIR/doc
        mkdir -p $MYDIR/doc_cn
        mkdir -p $PADDLE_SOURCE_DIR/build_doc
        pushd $PADDLE_SOURCE_DIR/build_doc
        cmake .. -DWITH_DOC=ON
        make paddle_docs paddle_docs_cn
        cp -r $PADDLE_SOURCE_DIR/build_doc/doc/en/html/* $MYDIR/doc
        cp -r $PADDLE_SOURCE_DIR/build_doc/doc/cn/html/* $MYDIR/doc_cn
        popd
        rm -rf $PADDLE_SOURCE_DIR/build_doc
        ;;
    "--help")
        usage
        ;;
    *)
        usage
        ;;
esac
