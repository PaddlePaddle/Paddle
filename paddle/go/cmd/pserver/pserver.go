package main

import (
	"flag"
	"net"
	"net/http"
	"net/rpc"
	"strconv"

	"github.com/PaddlePaddle/Paddle/paddle/go/pserver"
)

func main() {
	port := flag.Int("p", 0, "port of the pserver")
	flag.Parse()

	s := pserver.NewService()
	err := rpc.Register(s)
	if err != nil {
		panic(err)
	}

	rpc.HandleHTTP()
	l, err := net.Listen("tcp", ":"+strconv.Itoa(*port))
	if err != nil {
		panic(err)
	}

	err = http.Serve(l, nil)
	if err != nil {
		panic(err)
	}
}
