package master

import (
	"log"
	"time"

	"github.com/PaddlePaddle/Paddle/go/connection"
)

// Addresser provide the address of the master server.
type Addresser interface {
	Address() string
}

// Client is the client of the master server.
type Client struct {
	conn *connection.Conn
}

// NewClient creates a new Client.
func NewClient(addr Addresser) *Client {
	c := &Client{}
	c.conn = connection.New()
	go c.monitorMaster(addr)
	return c
}

func (c *Client) monitorMaster(addr Addresser) {
	lastMaster := ""
	monitor := func() {
		curMaster := addr.Address()
		if curMaster != lastMaster {
			if curMaster == "" {
				err := c.conn.Close()
				if err != nil {
					log.Println(err)
				}
			} else {
				err := c.conn.Connect(curMaster)
				if err != nil {
					log.Println(err)

					// connect to addr failed, set
					// to last known addr in order
					// to retry next time.
					curMaster = lastMaster
				}

			}
		}

		lastMaster = curMaster
	}

	monitor()
	ticker := time.NewTicker(10 * time.Second)
	for _ = range ticker.C {
		monitor()
	}
}

// GetTask gets a new task from the master server.
func (c *Client) GetTask() (Task, error) {
	var dummy int
	var t Task
	err := c.conn.Call("Service.GetTask", dummy, &t)
	return t, err
}

// TaskFinished tells the master server a task is finished.
func (c *Client) TaskFinished(taskID int) error {
	var dummy int
	return c.conn.Call("Service.TaskFinished", taskID, &dummy)
}
