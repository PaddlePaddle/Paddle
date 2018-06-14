// Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

// http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package master

import "sync"

// InMemStore is an in memory implementation of Store interface.
//
// It does not tolerate the fault that causes the program to crash.
type InMemStore struct {
	mu  sync.Mutex
	buf []byte
}

// Save saves the state into the in-memory store.
func (m *InMemStore) Save(state []byte) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.buf = state
	return nil
}

// Load loads the state from the in-memory store.
func (m *InMemStore) Load() ([]byte, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	return m.buf, nil
}

// Shutdown shuts down the in mem store.
func (m *InMemStore) Shutdown() error {
	return nil
}
