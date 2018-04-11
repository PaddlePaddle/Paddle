# PaddlePaddle Releasing Process

PaddlePaddle manages its branches using "git-flow branching model", and [Semantic Versioning](http://semver.org/) as it's version number semantics.

Each time we release a new PaddlePaddle version, we should follow the below steps:

1. Fork a new branch from `develop` named `release/[version]`, e.g. `release/0.10.0`.
1. Push a new tag on the release branch, the tag name should be like `[version]rc.patch`. The
   first tag should be `0.10.0rc1`, and the second should be `0.10.0.rc2` and so on.
1. After that, we should do:
  * Run all regression test on the Regression Test List (see PaddlePaddle TeamCity CI), to confirm
      that this release has no major bugs.
        * If regression test fails, we must fix those bugs and create a new `release/[version]`
          branch from previous release branch.
    * Modify `python/setup.py.in`, change the version number and change `ISTAGED` to `True`.
    * Publish PaddlePaddle release wheel packages to pypi (see below instructions for detail).
    * Update the Docker images (see below instructions for detail).
1. After above step, merge `release/[version]` branch to master and push a tag on the master commit,
   then merge `master` to `develop`.
1. Update the Release Note.          

***NOTE:***

* Do ***NOT*** merge commits from develop branch to release branches to keep the release branch contain
  features only for current release, so that we can test on that version.
* If we want to fix bugs on release branches, we must merge the fix to master, develop and release branch.

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

We use [git-flow](http://nvie.com/posts/a-successful-git-branching-model/) as our branching model,
with some modifications:

* `master` branch is the stable branch. Each version on the master branch is tested and guaranteed.
* `develop` branch is for development. Each commit on develop branch has passed CI unit test, but no
  regression tests are run.
* `release/[version]` branch is used to publish each release. Latest release version branches have
  bugfix only for that version, but no feature updates.
* Developer forks are not required to follow
  [git-flow](http://nvie.com/posts/a-successful-git-branching-model/)
  branching model, all forks is like a feature branch.
    * Advise: developer fork's develop branch is used to sync up with main repo's develop branch.
    * Advise: developer use it's fork's develop branch to for new branch to start developing.
  * Use that branch on developer's fork to create pull requests and start reviews.
      * developer can push new commits to that branch when the pull request is open.
* Bug fixes are also started from developers forked repo. And, bug fixes branch can merge to
  `master`, `develop` and `releases`.

## PaddlePaddle Regression Test List

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
