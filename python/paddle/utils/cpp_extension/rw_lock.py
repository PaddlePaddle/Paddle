# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import threading
"""
Simple Implement RWLock with threading.Lock. Once a thread has acquired a lock, 
subsequent attempts to acquire it block, until it is released; any thread may release it.

  - Read Lock:     Can only acquire the inner lock once no write thread or self._read_num == 1
  - Read Release:  Release the inner lock once self._read_num decrease into 0
  - Write Lock:    Can only acquire the inner lock once no other write or read thread hold the lock.
  - Write Release: Release the lock instantly
"""


class RWLock(object):
    def __init__(self):
        # used for Read / Write
        self._lock = threading.Lock()
        # used to protect updating self._read_num
        self._extra_lock = threading.Lock()
        self._read_num = 0

    def read_acquire(self):
        """
        Acquire the lock, block current thread until getting it successfully.
        """
        with self._extra_lock:
            self._read_num += 1
            if self._read_num == 1:
                self._lock.acquire(blocking=True)

    def read_release(self):
        """
        Release the lock. This can be called from any thread, not only the thread which has acquired the lock.
        """
        with self._extra_lock:
            self._read_num -= 1
            if self._read_num == 0:
                self._lock.release()

    def write_acquire(self):
        """
        Acquire the lock, block current thread until getting it successfully.
        """
        self._lock.acquire(blocking=True)

    def write_release(self):
        """
        Release the lock. This can be called from any thread, not only the thread which has acquired the lock.
        """
        self._lock.release()

    def locked(self):
        """
        Return true if the lock is acquired.
        """
        return self._lock.locked()


# Global singleton
RWLOCK = RWLock()
