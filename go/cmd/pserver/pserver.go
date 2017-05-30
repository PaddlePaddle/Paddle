package main

import (
	"net"
	"net/http"
	"net/rpc"
	"strconv"

	"github.com/namsral/flag"

	"github.com/PaddlePaddle/Paddle/go/pserver"
)

func main() {
	port := flag.Int("port", 0, "port of the pserver")
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
