include(external/cpm)

# This will automatically clone CCCL from GitHub and make the exported cmake targets available
cpmaddpackage(
  NAME CCCL GITHUB_REPOSITORY nvidia/cccl GIT_TAG
  main # Fetches the latest commit on the main branch
)
