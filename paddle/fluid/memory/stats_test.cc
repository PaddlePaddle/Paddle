// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/memory/stats.h"
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "gtest/gtest.h"

namespace paddle {
namespace memory {

TEST(stats_test, MultiThreadReadWriteTest) {
  std::string stat_type = "Allocated";
  size_t thread_num = 3;
  size_t data_num = 10;

  std::condition_variable cv;
  std::mutex mutex;
  std::vector<std::thread> threads;
  size_t ready_thread_num = 0;

  for (size_t i = 0; i < thread_num; ++i) {
    threads.emplace_back(
        [&stat_type, data_num, &cv, &mutex, &ready_thread_num]() {
          for (size_t data = 0; data < data_num; ++data) {
            StatUpdate(stat_type, 0, data);
          }
          /* lock guard*/ {
            std::lock_guard<std::mutex> lock_guard{mutex};
            ++ready_thread_num;
            cv.notify_one();
          }
          // Sleep here to not exit before the main thread checking stat
          // results, because the thread-local stat data will be destroyed when
          // the thread exit
          std::this_thread::sleep_for(std::chrono::seconds(1));
        });
  }

  std::unique_lock<std::mutex> unique_lock(mutex);
  cv.wait(unique_lock, [&ready_thread_num, thread_num]() {
    return ready_thread_num == thread_num;
  });

  EXPECT_EQ(StatGetCurrentValue(stat_type, 0),
            int64_t((thread_num * data_num * (data_num - 1)) >> 1));

  for (size_t i = 0; i < thread_num; ++i) {
    threads[i].join();
  }
}

TEST(stats_test, PeakValueTest) {
  std::string stat_type = "Allocated";
  std::vector<int64_t> datas = {
      543149808935355, 634698327471328, 706215795436611, 577939367795333,
      419479490054362, 21975227714595,  812939817942250, 984428837942082,
      537304104446806, 685008544452453, 563352858161268, 690143831596330,
      964829938186077, 476984078018245, 804403365180177, -57918691189304,
      947611269236893, 752188963801927, 710946451346683, -49226452527666,
      -59049377393968, 14128239868858,  463298869064035, 71954818131880,
      894087341752481, 971337322257029, 202325222441382, 128423535063606,
      -89146949094815, 756429151957759, 444400180007032, 937040878834954,
      303916192293233, 16628488962638,  544031750807065, 392396591234910,
      686663859558365, 941126625484539, 120719755546781, 938838399629825,
      364952832531949, 237865770815218, -64409925441421, 130095171433100,
      140906146174023, 635514857321759, -65954585142544, 505369882354612,
      939334896592688, 591590196329715, 424834428510773, 316569328289240,
      44932622352645,  464924685290752, 396541464249293, 937169087747437,
      437992536948503, 44395833829426,  968496835801562, 80493658180301,
      836093264717766, 3339912102452,   -32067753603273, 77353521424986,
      290980283590981, 496135147814915, 335112580987207, 571094882208895,
      776581672377954, -83075504255716, -93690563747742, 115144063088100,
      422629490055299, 917988755593299, 283511671626409, 715179006446336,
      760708617230450, 183628659314298, 899792829140365, 214949068928854,
      848851506468080, 791041814082114, 801591030978388, 526551272394511,
      781034506085043, 279998089943681, 907197980150568, 974365521595836,
      282127262539024, 272870474932399, 346617645597508, 409964014011113,
      746465732805300, -74049761897414, -65640372433924, 852009039806484,
      305079802044257, -48409757869238, 266031781660228, 327287322379820};

  int64_t peak_value = ((int64_t)1) << 63;
  int64_t sum = 0;
  for (int64_t data : datas) {
    StatUpdate(stat_type, 0, data);
    sum += data;
    peak_value = std::max(peak_value, sum);
  }
  EXPECT_EQ(StatGetPeakValue(stat_type, 0), peak_value);
}

}  // namespace memory
}  // namespace paddle
