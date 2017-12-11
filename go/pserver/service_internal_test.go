package pserver

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

const testDir = "./test_data"

type myKV struct {
	m map[string][]byte
}

func (m *myKV) GetKey(key string, timeout time.Duration) ([]byte, error) {
	if m.m == nil {
		m.m = make(map[string][]byte)
	}
	return m.m[key], nil
}

func (m *myKV) PutKey(key string, value []byte, timeout time.Duration, withLease bool) error {
	if m.m == nil {
		m.m = make(map[string][]byte)
	}
	m.m[key] = value
	return nil
}

func TestCheckpoint(t *testing.T) {
	kv := &myKV{}
	s, err := NewService(0, time.Hour, testDir, kv, nil)
	assert.Nil(t, err)
	err = s.checkpoint()
	assert.Nil(t, err)
	_, err = LoadCheckpoint(kv, 0)
	assert.Nil(t, err)
}

func float32ToByte(f float32) []byte {
	var buf bytes.Buffer
	err := binary.Write(&buf, binary.LittleEndian, f)
	if err != nil {
		fmt.Println("binary.Write failed:", err)
	}
	return buf.Bytes()
}

func TestCheckpointWithData(t *testing.T) {
	kv := &myKV{}
	s, err := NewService(0, time.Hour, testDir, kv, nil)
	assert.Nil(t, err)

	var content []byte
	for i := 0; i < 50000; i++ {
		content = append(content, float32ToByte(float32(i))...)
	}

	p1 := Parameter{Name: "p1", ElementType: 1, Content: content}
	err = s.InitParam(ParameterWithConfig{Param: p1}, nil)
	assert.Nil(t, err)

	err = s.FinishInitParams(0, nil)
	assert.Nil(t, err)

	var p2 Parameter
	err = s.GetParam(p1.Name, &p2)
	assert.Nil(t, err)
	assert.Equal(t, p1, p2)

	err = s.checkpoint()
	assert.Nil(t, err)
	cp, err := LoadCheckpoint(kv, 0)
	assert.Nil(t, err)
	s1, err := NewService(0, time.Hour, testDir, kv, cp)
	assert.Nil(t, err)

	var p3 Parameter
	err = s1.GetParam(p1.Name, &p3)
	assert.Nil(t, err)
	assert.Equal(t, p1, p3)
}
