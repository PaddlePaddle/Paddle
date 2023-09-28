include(external/cpm)

# This will automatically clone CCCL from GitHub and make the exported cmake targets available
cpmaddpackage(
  NAME
  CCCL
  GITHUB_REPOSITORY
  nvidia/cccl
  GIT_TAG
  1f6e4bcae0fbf1bbed87f88544d8d2161c490fc1 # The latest commit has bugs in windows, so we set a fix commit.
)
