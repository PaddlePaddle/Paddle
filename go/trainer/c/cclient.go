package main

/*
#include <stdio.h>

#define PADDLE_TRAINER_INIT_ERROR -1
typedef int paddle_trainer;

*/
import "C"
import (
	"sync"

	"github.com/PaddlePaddle/Paddle/go/trainer"
)

var mu sync.Mutex
var handleMap = make(map[C.paddle_trainer]*trainer.Trainer)
var curHandle C.paddle_pserver_client

func add(t *trainer.Trainer) C.paddle_trainer {
	mu.Lock()
	defer mu.Unlock()
	instance := curHandle
	curHandle++
	handleMap[instance] = t
	return trainer
}

//export paddle_new_trainer
func paddle_new_trainer(endpoints *C.char) C.paddle_trainer {
	t, err := trainer.NewTrainer(C.GoString(endpoints))
	if err != nil {
		return C.PADDLE_TRAINER_ERROR
	}
	return add(t)
}

func main() {}
