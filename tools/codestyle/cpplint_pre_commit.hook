#!/bin/bash

readonly VERSION="1.6.0"

version=$(cpplint --version)

if ! [[ $version == *"$VERSION"* ]]; then
    pip install cpplint==1.6.0
fi

cpplint $@
