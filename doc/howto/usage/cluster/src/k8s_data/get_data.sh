#!/bin/sh

out_dir=$OUT_DIR
split_count=$SPLIT_COUNT

set -e

mkdir -p $out_dir
cp -r /quick_start $out_dir/

mkdir -p $out_dir/0/data
cd $out_dir/0/data
wget http://paddlepaddle.bj.bcebos.com/demo/quick_start_preprocessed_data/preprocessed_data.tar.gz
tar zxvf preprocessed_data.tar.gz
rm preprocessed_data.tar.gz

split -d --number=l/$split_count -a 5 train.txt train.
mv train.00000 train.txt

cd $out_dir
end=$(expr $split_count - 1)
for i in $(seq 1 $end); do
    mkdir -p $i/data
    cp -r 0/data/* $i/data
    mv $i/data/train.`printf %05d $i` $i/data/train.txt
done;
