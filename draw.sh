#!/bin/bash -ex

dot_files=$(ls $PWD/*.dot)

for file in $dot_files; do
    # echo $file
    ext=${file##*.}
    # echo $ext
    fname=$(basename $file ".$ext")
    # echo $fname
    dot -Tpdf $file -o "$PWD/$fname.pdf"
done

