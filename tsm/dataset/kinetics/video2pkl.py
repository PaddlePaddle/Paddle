#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import os
import sys
import glob
try:
    import cPickle as pickle
except:
    import pickle
from multiprocessing import Pool

# example command line: python generate_k400_pkl.py kinetics-400_train.csv 8
# 
# kinetics-400_train.csv is the training set file of K400 official release
# each line contains laebl,youtube_id,time_start,time_end,split,is_cc

assert (len(sys.argv) == 5)

f = open(sys.argv[1])
source_dir = sys.argv[2]
target_dir = sys.argv[3]
num_threads = sys.argv[4]
all_video_entries = [x.strip().split(',') for x in f.readlines()]
all_video_entries = all_video_entries[1:]
f.close()

category_label_map = {}
f = open('kinetics400_label.txt')
for line in f:
    ens = line.strip().split(' ')
    category = " ".join(ens[0:-1])
    label = int(ens[-1])
    category_label_map[category] = label
f.close()


def generate_pkl(entry):
    mode = entry[4]
    category = entry[0].strip('"')
    category_dir = category
    video_path = os.path.join(
        './',
        entry[1] + "_%06d" % int(entry[2]) + "_%06d" % int(entry[3]) + ".mp4")
    video_path = os.path.join(source_dir, category_dir, video_path)
    label = category_label_map[category]

    vid = './' + video_path.split('/')[-1].split('.')[0]
    if os.path.exists(video_path):
        if not os.path.exists(vid):
            os.makedirs(vid)
        os.system('ffmpeg -i ' + video_path + ' -q 0 ' + vid + '/%06d.jpg')
    else:
        print("File not exists {}".format(video_path))
        return

    images = sorted(glob.glob(vid + '/*.jpg'))
    ims = []
    for img in images:
        f = open(img, 'rb')
        ims.append(f.read())
        f.close()

    output_pkl = vid + ".pkl"
    output_pkl = os.path.join(target_dir, output_pkl)
    f = open(output_pkl, 'wb')
    pickle.dump((vid, label, ims), f, protocol=2)
    f.close()

    os.system('rm -rf %s' % vid)


pool = Pool(processes=int(sys.argv[4]))
pool.map(generate_pkl, all_video_entries)
pool.close()
pool.join()
