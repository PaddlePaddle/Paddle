# Design Doc: `go_{library,binary,test}`

## Concerns

1. Need to build Go libraries callable from Go and from C.

   For usual Go libraries, the bulding command line is as

   ```bash
   go build foo.go bar.go -o libfoobar.a
   ```

   For Go libraries callable from C/C++, the command line is

   ```bash
   go build -buildmode=c-archive foo.go bar.go -o libstatic.a
   ```

   or for shared libraries:

   ```bash
   go build -buildmode=c-shared foo.go bar.go -o libdynamic.so
   ```

   and `foo.go`, `bar.go`, etc must start with a line `package main`,
   which defines all symbols in special pacakge `main`.  There also
   must be a `func main`, which could have empty body.

1. Need to support building-in-Docker.

   We are going to support two ways to building -- (1) in Ubuntu, and
   (2) in Docker container whose base image is Ubuntu.

   The challenge is (2), because to build in Docker, we run the
   development image as:

   ```bash
   git clone https://github.com/PaddlePaddle/Paddle -o paddle
   cd paddle
   docker run -v $PWD:/paddle paddle:dev
   ```

   which maps the local repo to `/paddle` in the container.

   This assumes that all Go code, including third-party packages, must
   be in the local repo.  Actually, it assumes that `$GOPATH` must be
   in the local repo.  This would affect how we write `import`
   statemetns, and how we maintain third-party packages.


## A Solution

This might not be the optimal solution.  Comments are welcome.

### Directory structure

We propose the following directory structure:

```
https://github.com/PaddlePaddle/Paddle
                                  ↓ git clone
                         ~/work/paddle/go/pkg1/foo.go
                                         /pkg2/bar.go
                                         /cmd/cmd1/wee.go
                                         /cmd/cmd2/qux.go
                                         /github.com/someone/a_3rd_party_pkg (Git submodule to a 3rd-party pkg)
                                         /golang.org/another_3rd_party_pkg (Git submodule to another one)
                                      /build/go ($GOPATH, required by Go toolchain)
                                               /src (symlink to ~/work/paddle/go/)
                                               /pkg (libraries built by Go toolchain)
                                               /bin (binaries bulit by Go toolchain)
```

Above figure explains how we organize Paddle's Go code:

1. Go source code in Paddle is in `/go` of the repo.
1. Each library package is a sub-directory under `/go`.
1. Each (executable) binary package is a sub-directory under
   `/go/cmd`.  This is the source tree convention of Go itself.
1. Each 3rd-party Go package is a Git submodule under `/go`.

These rules make sure that all Go source code are in `/go`.

At build-time, Go toolchain requires a directory structure rooted at
`$GOPATH` and having three sub-directories: `$GOPATH/src`,
`$GOPATH/pkg`, and `$GOPATH/bin`, where `$GOPATH/src` is the source
tree and the root of Go's `import` paths.

For example, if `/go/pkg1/foo.go` contains `import
"github.com/someone/a_3rd_party_pkg"`, the Go toolchain will find the
package at `$GOPATH/src/github.com/someone/a_3rd_party_pkg`.

In order to create such a `$GOPATH`, our build system creates
`/build/go`.  Remeber that we want to make sure that all output files
generated at build-time are place in `/build`.

Under `/build/go`, our build system creates a symoblic link `src`
pointing to `/go`, where all Go source code resides.
