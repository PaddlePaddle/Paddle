/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifndef HL_TIME_H_
#define HL_TIME_H_
#include <cstdint>
/**
 * @brief   High resolution timer.
 *
 * @return  int64_t the representation value of the object as a
 *                  count of periods, which are not necessarily
 *                  seconds.
 *
 * @note    It is used to generate random perturbation parameters.
 */
int64_t getCurrentTimeStick(void);

#endif /* HL_TIME_H_ */
