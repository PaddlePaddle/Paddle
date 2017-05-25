package master

import (
	"errors"
	"log"
	"sync"
	"time"

	"github.com/PaddlePaddle/Paddle/paddle/go/recordio"
)

const (
	targetTaskCount = 300
)

// errors
var (
	ErrNoMoreTask          = errors.New("no more task for current pass")
	ErrPendingTaskNotFound = errors.New("pending task not found")
)

// Service is the master server service.
type Service struct {
	timeoutDur time.Duration
	timeoutMax int

	mu         sync.Mutex
	taskQueues taskQueues
}

// Recover recovers service state from etcd.
func Recover() (*Service, error) {
	// TODO(helin): recover from snapshot state from etcd.
	return nil, nil
}

func partition(chunks []Chunk, targetTaskCount int) []taskEntry {
	id := 0
	chunkPerTask := len(chunks) / targetTaskCount
	if chunkPerTask <= 0 {
		chunkPerTask = 1
	}

	var result []taskEntry
	var cur taskEntry
	for i, c := range chunks {
		if i%chunkPerTask == 0 && len(cur.Task.Chunks) > 0 {
			cur.Task.ID = id
			id++
			result = append(result, cur)
			cur.Task.Chunks = nil
		}

		cur.Task.Chunks = append(cur.Task.Chunks, c)
	}

	if len(cur.Task.Chunks) > 0 {
		cur.Task.ID = id
		id++
		result = append(result, cur)
	}

	return result
}

// NewService creates a new service.
func NewService(chunks []Chunk, timeoutDur time.Duration, timeoutMax int) *Service {
	s := &Service{}
	s.timeoutDur = timeoutDur
	s.timeoutMax = timeoutMax
	s.taskQueues = taskQueues{}
	s.taskQueues.Pending = make(map[int]taskEntry)
	s.taskQueues.Todo = partition(chunks, targetTaskCount)
	return s
}

// Chunk is a chunk of data consisted of several data instances.
type Chunk struct {
	Idx   int // index of the chunk within the file
	Path  string
	Index recordio.Index // block index
}

// Task is the basic unit of data instances assigned to trainers.
type Task struct {
	ID     int
	Chunks []Chunk
}

type taskEntry struct {
	Epoch      int
	NumTimeout int
	Task       Task
}

type taskQueues struct {
	Todo    []taskEntry
	Pending map[int]taskEntry // map from task ID to task entry
	Done    []taskEntry
	Failed  []Task
}

// *must* be called with s.mu being held.
func (s *Service) snapshot() error {
	// TODO(helin): snapshot state on etcd.
	return nil
}

// GetTask gets a new task from the service.
func (s *Service) GetTask(dummy int, task *Task) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if len(s.taskQueues.Todo) == 0 {
		return ErrNoMoreTask
	}

	t := s.taskQueues.Todo[0]
	t.Epoch++
	s.taskQueues.Todo = s.taskQueues.Todo[1:]
	s.taskQueues.Pending[t.Task.ID] = t
	err := s.snapshot()
	if err != nil {
		return err
	}

	time.AfterFunc(s.timeoutDur, func(taskID int, epoch int) func() {
		return func() {
			s.mu.Lock()
			defer s.mu.Unlock()

			t, ok := s.taskQueues.Pending[taskID]
			if !ok {
				return
			}

			if t.Epoch != epoch {
				// new epoch, task launched after the
				// schedule of this timeout check.
				return
			}

			defer func() {
				err := s.snapshot()
				if err != nil {
					log.Println(err)
				}
			}()

			delete(s.taskQueues.Pending, t.Task.ID)

			t.NumTimeout++
			if t.NumTimeout > s.timeoutMax {
				s.taskQueues.Failed = append(s.taskQueues.Failed, t.Task)
				return
			}

			s.taskQueues.Todo = append(s.taskQueues.Todo, t)
		}
	}(t.Task.ID, t.Epoch))
	return nil
}

// TaskFinished tell the service that a task is finished.
func (s *Service) TaskFinished(taskID int, dummy *int) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	t, ok := s.taskQueues.Pending[taskID]
	if !ok {
		return ErrPendingTaskNotFound
	}

	// task finished, reset timeout
	t.NumTimeout = 0
	s.taskQueues.Done = append(s.taskQueues.Done, t)
	delete(s.taskQueues.Pending, taskID)
	return s.snapshot()
}
