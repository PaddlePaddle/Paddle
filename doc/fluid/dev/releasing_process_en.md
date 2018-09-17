# PaddlePaddle Releasing Process

PaddlePaddle manages its branches using "git-flow branching model", and [Semantic Versioning](http://semver.org/) as it's version number semantics.

Each time we release a new PaddlePaddle version, we should follow the below steps:

1. Create a new release branch from `develop`，named `release/[version]`. E.g.，`release/0.10.0`
2. Create a new tag for the release branch, tag format: `version-rc.Patch`. E.g. the first tag is `0.10.0-rc0`。
3. New release branch normally doesn't accept new features or optimizations. QA will test on the release branch. Developer should develop based on `develop` branch.
4. If QA or Developer find bugs. They should first fix and verify on `develop` branch. Then cherry-pick the fix to the release branch. Wait until the release branch is stable.
5. If necessary, create a new tag on the relese branch, e.g. `0.10.0-rc1`. Involve more users to try it and repeat step 3-4.
6. After release branch is stable，Create the official release tag，such as `0.10.0`.
7. Release the python wheel package to pypi.
8. Update the docker image (More details below).

NOTE:

* bug fix should happen on `develop` branch, then cherry-pick to relese branch. Avoid developing directly on release branch.

* release normally only accept bug fixes. Don't add new features.


## Publish Wheel Packages to pypi

1. Use our [CI tool](https://paddleci.ngrok.io/project.html?projectId=Manylinux1&tab=projectOverview)
   to build all wheel packages needed to publish. As shown in the following picture, choose a build
     version, click "..." button on the right side of "Run" button, and switch to the second tab in the
pop-up box, choose the current release branch and click "Run Build" button. You may repeat this
     step to start different versions of builds.
    <img src="https://raw.githubusercontent.com/PaddlePaddle/Paddle/develop/doc/fluid/images/ci_build_whl.png">
1. After the build succeeds, download the outputs under "Artifacts" including capi, `cp27m` and `cp27mu`.
1. Since pypi.python.org follows [PEP 513](https://www.python.org/dev/peps/pep-0513), before we
     upload the package using `twine`, we need to rename the package from `linux_x86_64` to
     `manylinux1_x86_64`.
1. Start the upload:
     ```
     cd build/python
     pip install twine
     twine upload dist/[package to upload]
     ```

* NOTE: We use a special Docker image to build our releases to support more Linux distributions, you can
  download it from https://hub.docker.com/r/paddlepaddle/paddle_manylinux_devel/tags/, or build it using
    scripts under `tools/manylinux1`.
* pypi does not allow overwrite the already uploaded version of wheel package, even if you delete the
  old version. you must change the version number before upload a new one.

### Publish wheel Packages for MacOS

You need to build the binary wheel package for MacOS before publishing, to
make sure that the package can be used by many versions of MacOS
(10.11, 10.12, 10.13) and different python installs (python.org, homebrew, etc.),
you must build the package ***exactly*** following below steps:

Build steps:

1. install python from python.org downloads, and make sure it's currently in use
   in your system.
1. `export MACOSX_DEPLOYMENT_TARGET=10.11`, use `10.11` is enough for recent versions.
1. `git clone https://github.com/PaddlePaddle/Paddle.git && cd Paddle && mkdir build && cd build`
1. `cmake -DWITH_GPU=OFF -DWITH_MKL=OFF -DWITH_SYSTEM_BLAS=OFF  ..`, make sure the output of `cmake` command is using the correct python interpreter installed from python.org
1. `make -j`
1. `pip install delocate`
1. `mkdir fixed_wheel && delocate-wheel -w fixed_wheel python/dist/*.whl`

Then the whl under `fixed_wheel` is ready to upload.

Install steps:

1. run `pip install paddlepaddle...whl`
1. find the `libpython.dylib` that are currently in use:
    - for python.org package installs, do nothing.
    - for other python installs, find the path of `libpython*.dylib` and `export LD_LIBRARY_PATH=you path && DYLD_LIBRARY_PATH=your path`

## Publish Docker Images

Our CI tool will push latest images to DockerHub, so we only need to push a version tag like:

```
docker pull [image]:latest
docker tag [image]:latest [image]:[version]
docker push [image]:[version]
```

Tags that need to be updated are:
* `[version]`: CPU only version image
* `[version]-openblas`: openblas version image
* `[version]-gpu`: GPU version（using CUDA 8.0 cudnn 5）
* `[version]-gpu-[cudaver]-[cudnnver]`: tag for different cuda, cudnn versions

You can then checkout the latest pushed tags at https://hub.docker.com/r/paddlepaddle/paddle/tags/.

## Branching Model

PaddlePaddle uses [Trunk Based Development](https://trunkbaseddevelopment.com/) as our branching model.

* `develop` branch is used for development. Each comment to `develop` branc goes through unit tests and model regression tests.
* `release/[version]` branch is used for each release. Release branch is used for tests, bug fix and evetual release.
* `master` branch as been deprecated for historical reasons

* Developer's feature branch。
	* Developer's feature branch should sync with upstream `develop` branch.
	* Developer's feature branch should be forked from upstream `develop` branch.
	* After feature branch is ready, create a `Pull Request` against the Paddle repo and go through code review.
	   * In the review process, develop modify codes and push to their own feature branch.

## PaddlePaddle Regression Test List

TODO

### All Chapters of PaddlePaddle Book

We need to guarantee that all the chapters of PaddlePaddle Book can run correctly. Including
V1 (`paddle_trainer` training) and V2 training and Fluid training.

<table>
<thead>
<tr>
<th></th>
<th>Linear Regression</th>
<th>Recognize Digits</th>
<th>Image Classification</th>
<th>Word2Vec</th>
<th>Personalized Recommendation</th>
<th>Sentiment Analysis</th>
<th>Semantic Role Labeling</th>
<th>Machine Translation</th>
</tr>
</thead>

<tbody>
<tr>
<td>API.V2 + Docker + GPU </td>
<td>  </td>
<td> </td>
<td>  </td>
<td> </td>
<td>  </td>
<td> </td>
<td>  </td>
<td> </td>
</tr>

<tr>
<td> API.V2 + Docker + CPU </td>
<td>  </td>
<td> </td>
<td>  </td>
<td> </td>
<td>  </td>
<td> </td>
<td>  </td>
<td> </td>
</tr>

<tr>
<td>`paddle_trainer` + Docker + GPU </td>
<td>  </td>
<td> </td>
<td>  </td>
<td> </td>
<td>  </td>
<td> </td>
<td>  </td>
<td> </td>
</tr>

<tr>
<td>`paddle_trainer` + Docker + CPU </td>
<td>  </td>
<td> </td>
<td>  </td>
<td> </td>
<td>  </td>
<td> </td>
<td>  </td>
<td> </td>
</tr>

<tr>
<td> API.V2 + Ubuntu + GPU</td>
<td>  </td>
<td> </td>
<td>  </td>
<td> </td>
<td>  </td>
<td> </td>
<td>  </td>
<td> </td>
</tr>

<tr>
<td>API.V2 + Ubuntu + CPU </td>
<td>  </td>
<td> </td>
<td>  </td>
<td> </td>
<td>  </td>
<td> </td>
<td>  </td>
<td> </td>
</tr>

<tr>
<td> `paddle_trainer` + Ubuntu + GPU</td>
<td>  </td>
<td> </td>
<td>  </td>
<td> </td>
<td>  </td>
<td> </td>
<td>  </td>
<td> </td>
</tr>

<tr>
<td> `paddle_trainer` + Ubuntu + CPU</td>
<td>  </td>
<td> </td>
<td>  </td>
<td> </td>
<td>  </td>
<td> </td>
<td>  </td>
<td> </td>
</tr>
</tbody>
</table>
