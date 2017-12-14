#!/usr/bin/env sh
# This script downloads the mnist data and unzips it.
set -e
DIR="$( cd "$(dirname "$0")" ; pwd -P )"
rm -rf "$DIR/mnist_data"
mkdir "$DIR/mnist_data"
cd "$DIR/mnist_data"

echo "Downloading..."

for fname in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte
do
    if [ ! -e $fname ]; then
        wget --no-check-certificate http://yann.lecun.com/exdb/mnist/${fname}.gz
        gunzip ${fname}.gz
    fi
done
