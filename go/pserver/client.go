package pserver

import (
	"errors"
	"hash/fnv"
	"sort"
	"time"

	"github.com/PaddlePaddle/Paddle/go/connection"
	log "github.com/sirupsen/logrus"
)

// TODO(helin): add RPC call retry logic

// Selector selects if the client should initialize parameter servers.
type Selector interface {
	Select() bool
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
					log.Errorln(err)
				}

				continue
			}

			err := c.pservers[i].Connect(curServers[i].Addr)
			if err != nil {
				log.Errorln(err)

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
func (c *Client) BeginInitParams() bool {
	return c.sel.Select()
}

// InitParam initializes the parameter on parameter servers.
func (c *Client) InitParam(paramWithConfigs ParameterWithConfig) error {
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
	return nil
}

// SendGrads sends gradients to parameter servers for updating
// parameters.
func (c *Client) SendGrads(grads []Gradient) error {
	if len(grads) == 0 {
		return errors.New("no gradient received")
	}
	errCh := make(chan error, len(grads))
	for _, g := range grads {
		go func(g Gradient) {
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
	param Parameter
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
func (c *Client) GetParams(names []string) ([]Parameter, error) {
	rCh := make(chan result, len(names))

	for idx, name := range names {
		go func(name string, idx int) {
			var parameter Parameter
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

	ps := make([]Parameter, len(rs))
	for i := range rs {
		ps[i] = rs[i].param
	}

	return ps, nil
}

// Save indicates parameters to save the parameter to the given path.
func (c *Client) Save(path string) error {
	errCh := make(chan error, len(c.pservers))

	for _, p := range c.pservers {
		err := p.Call("Service.Save", path, nil)
		errCh <- err
	}

	recv := 0
	for err := range errCh {
		if err != nil {
			return err
		}

		recv++
		if recv == len(c.pservers) {
			break
		}
	}

	// TODO(helin): there will be many files under path, need to
	// merge them into a single file.
	return nil
}

func strHash(s string) uint32 {
	h := fnv.New32a()
	h.Write([]byte(s))
	return h.Sum32()
}

// TODO(helin): now partition only select which parameter server to
// send the entire parameter. We need to partition a parameter into
// small blocks and send to different parameter servers.
func (c *Client) partition(key string) int {
	return int(strHash(key) % uint32(len(c.pservers)))
}
