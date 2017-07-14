package master

import "sync"

// InMemStore is an in memory implementation of Store interface.
//
// It does not tolerate the fault that casues the program to crash.
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
