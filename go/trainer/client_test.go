package trainer

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestInvaliedEtcdEndpoints(t *testing.T) {
	trainer := NewTrainer("localhost:12345", 5*time.Second)
	err := trainer.Init()
	assert.NotNil(t, err, "Invalid etcd endpoints should be a nil client")
}
