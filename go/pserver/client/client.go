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

package client

import (
	"errors"
	"hash/fnv"
	"sort"
	"time"

	"github.com/PaddlePaddle/Paddle/go/connection"
	"github.com/PaddlePaddle/Paddle/go/pserver"
	log "github.com/inconshreveable/log15"
)

// TODO(helin): add RPC call retry logic

// Selector selects if the client should initialize parameters and
// reports the initialization process done.
type Selector interface {
	// Select selects if the client should initialize parameter servers.
	Select() (bool, error)
	// Done indicates the initialization process is done.
	Done() error
}

// Server is the identification of a parameter Server.
type Server struct {
	Index int
	Addr  string
}

// Lister lists currently available parameter servers.
type Lister interface {
	List() []Server
}

// Client is the client to parameter servers.
type Client struct {
	sel      Selector
	pservers []*connection.Conn
}

// NewClient creates a new client.
func NewClient(l Lister, pserverNum int, sel Selector) *Client {
	c := &Client{sel: sel}
	c.pservers = make([]*connection.Conn, pserverNum)
	for i := 0; i < pserverNum; i++ {
		c.pservers[i] = connection.New()
	}
	go c.monitorPservers(l, pserverNum)
	return c
}

// monitorPservers monitors pserver addresses, and updates connection
// when the address changes.
func (c *Client) monitorPservers(l Lister, pserverNum int) {
	lastServers := make([]Server, pserverNum)
	ticker := time.NewTicker(10 * time.Second)
	monitor := func() {
		curServers := make([]Server, pserverNum)
		list := l.List()
		for _, l := range list {
			curServers[l.Index] = l
		}

		for i := range lastServers {
			if lastServers[i].Addr == curServers[i].Addr {
				continue
			}

			if curServers[i].Addr == "" {
				err := c.pservers[i].Close()
				if err != nil {
					log.Error("error closing connection to pserver", log.Ctx{"error": err})
				}

				continue
			}

			err := c.pservers[i].Connect(curServers[i].Addr)
			if err != nil {
				log.Error("error connecting to pserver", log.Ctx{"error": err})

				// connect to addr failed, set
				// to last known addr in order
				// to retry next time.
				curServers[i].Addr = lastServers[i].Addr
			}

		}

		lastServers = curServers
	}

	monitor()
	for range ticker.C {
		monitor()
	}
}

// BeginInitParams begins to initialize parameters on parameter
// servers.
//
// BeginInitParams will be called from multiple trainers, only one
// trainer will be selected to initialize the parameters on parameter
// servers. Other trainers will be blocked until the initialization is
// done, and they need to get the initialized parameters from
// parameter servers using GetParams.
func (c *Client) BeginInitParams() (bool, error) {
	return c.sel.Select()
}

// InitParam initializes the parameter on parameter servers.
func (c *Client) InitParam(paramWithConfigs pserver.ParameterWithConfig) error {
	return c.pservers[c.partition(paramWithConfigs.Param.Name)].Call("Service.InitParam", paramWithConfigs, nil)
}

// FinishInitParams tells parameter servers client has sent all
// parameters to parameter servers as initialization.
func (c *Client) FinishInitParams() error {
	for _, p := range c.pservers {
		err := p.Call("Service.FinishInitParams", 0, nil)
		if err != nil {
			return err
		}
	}
	return c.sel.Done()
}

// SendGrads sends gradients to parameter servers for updating
// parameters.
func (c *Client) SendGrads(grads []pserver.Gradient) error {
	if len(grads) == 0 {
		return errors.New("no gradient received")
	}
	errCh := make(chan error, len(grads))
	for _, g := range grads {
		go func(g pserver.Gradient) {
			err := c.pservers[c.partition(g.Name)].Call("Service.SendGrad", g, nil)
			errCh <- err
		}(g)
	}

	recv := 0
	for err := range errCh {
		if err != nil {
			return err
		}

		recv++
		if recv == len(grads) {
			break
		}
	}
	return nil
}

type result struct {
	idx   int
	param pserver.Parameter
	err   error
}

type results []result

func (r results) Len() int {
	return len(r)
}

func (r results) Less(i int, j int) bool {
	return r[i].idx < r[j].idx
}

func (r results) Swap(i int, j int) {
	r[i], r[j] = r[j], r[i]
}

// GetParams gets parameters from parameter servers.
func (c *Client) GetParams(names []string) ([]pserver.Parameter, error) {
	rCh := make(chan result, len(names))

	for idx, name := range names {
		go func(name string, idx int) {
			var parameter pserver.Parameter
			err := c.pservers[c.partition(name)].Call("Service.GetParam", name, &parameter)
			rCh <- result{idx: idx, param: parameter, err: err}
		}(name, idx)
	}

	var rs results
	recv := 0
	for r := range rCh {
		if r.err != nil {
			return nil, r.err
		}
		rs = append(rs, r)

		recv++
		if recv == len(names) {
			break
		}
	}
	sort.Sort(rs)

	ps := make([]pserver.Parameter, len(rs))
	for i := range rs {
		ps[i] = rs[i].param
	}

	return ps, nil
}

func strHash(s string) uint32 {
	h := fnv.New32a()
	_, _ = h.Write([]byte(s))
	return h.Sum32()
}

// TODO(helin): now partition only select which parameter server to
// send the entire parameter. We need to partition a parameter into
// small blocks and send to different parameter servers.
func (c *Client) partition(key string) int {
	return int(strHash(key) % uint32(len(c.pservers)))
}
