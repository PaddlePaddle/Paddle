package pserver

// ElementType is the type of elements of a Parameter.
type ElementType int

// Supported element types
const (
	Int32 ElementType = iota
	UInt32
	Int64
	UInt64
	Float32
	Float64
)

type Parameter struct {
	Name        string
	ElementType ElementType
	Content     []byte
}

type ParameterWithConfig struct {
	Param  Parameter
	Config []byte
}

type Gradient Parameter

type Client struct {
}

func NewClient(addr string) *Client {
	return &Client{}
}

func (c *Client) BeginInitParams(pserverConfigProto []byte) (bool, error) {
	return true, nil
}

func (c *Client) InitParam(paramWithConfigs ParameterWithConfig) error {
	return nil
}

func (c *Client) FinishInitParams() error {
	return nil
}

func (c *Client) SendGrads(grads []Gradient) error {
	return nil
}

func (c *Client) GetParams(names []string) ([]Parameter, error) {
	return nil, nil
}

func (c *Client) SaveModel(path string) error {
	return nil
}

func (c *Client) Cleanup() {
}
