# Parameter Server Side Request Handlers

Handlers classes under this directory is used for
parameter servers to deal with incomming requests
from trainers in background threads. The simple
scheme on server side is like:

```
forground             background
------------------------------------------
wait send barrier     handle send
    |                 handle send
    |                 handle send
    |                 handle send barrier
run graph             wait get barrier
    |                      |
    |                      |
end run graph         handle get
wait get barrier      handle get
    |                 handle get
    |                 handle get barrier
wait send barrier           |
------------------------------------------
```
