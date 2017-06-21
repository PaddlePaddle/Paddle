package main

/*
#include <stdio.h>

typedef int paddle_trainer;

*/
import "C"
import (
	"sync"
	"time"

	"github.com/PaddlePaddle/Paddle/go/trainer"
)

var mu sync.Mutex
var handleMap = make(map[C.paddle_trainer]*trainer.Trainer)
var curHandle C.paddle_trainer

func add(t *trainer.Trainer) C.paddle_trainer {
	mu.Lock()
	defer mu.Unlock()
	instance := curHandle
	curHandle++
	handleMap[instance] = t
	return instance
}

//export paddle_new_trainer
func paddle_new_trainer(endpoints *C.char, timeout C.int) C.paddle_trainer {
	t := trainer.NewTrainer(C.GoString(endpoints), time.Second*time.Duration(timeout))
	return add(t)
}

func main() {}
