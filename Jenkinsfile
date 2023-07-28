class PodTemplateFiles {
  private Map files = [
    'MThreads GPU': 'ci/templates/musa.yaml',
  ]

  public String getPodTemplateFile(String platform) {
    String file = files.get(platform)
    return file
  }
}

def ifTriggeredByTimer() {
  return currentBuild.getBuildCauses()[0].shortDescription == 'Started by timer'
}

pipeline {
  parameters {
    choice(name: 'HARDWARE_PLATFORM', choices: ['MThreads GPU'], description: 'Target hardware platform')
  }

  agent {
    kubernetes {
      yamlFile "${new PodTemplateFiles().getPodTemplateFile(params.HARDWARE_PLATFORM)}"
      defaultContainer "main"
    }
  }

  triggers {
    // UTC
    cron(env.BRANCH_NAME == 'main' ? '0 18 * * *' : '')
  }

  stages {
    stage("Codestyle-Check") {
      steps {
        container('main') {
          sh 'git config --global --add safe.directory \"*\"'
          sh '/bin/bash --login -c "BRANCH=origin/develop /bin/bash tools/codestyle/pre_commit.sh"'
          // sh 'git diff --name-only origin/develop..HEAD | xargs pre-commit run --files'
        }
      }
    }
    stage('Build') {
      steps {
        container('main') {
          sh 'echo Build'
          sh '/bin/bash --login -c "cp -r $PADDLE_MUSA_REPO_PATH/third_party/. third_party \
              && cp -r $PADDLE_MUSA_REPO_PATH/.git/modules .git"'
          sh '/bin/bash --login -c "ls -l && ls -l paddle"'
          // When installing paddle via whl within v0.1.1 docker there is an error:
          // paddlepaddle-0.0.0-cp37-cp37m-linux_x86_64.whl is not a supported wheel on this platform
          // the whl's name needs to be modified according to the whl suffix supported by pip.
          // sh '/bin/bash --login -c "PY_VERSION=3.8 /bin/bash paddle/scripts/paddle_build.sh build_only 64 && \
          //     find ./dist -name *whl | xargs pip install"'
          // sh '/bin/bash --login -c "MAX_JOBS=64 python setup.py install"'
        }
      }
    }
    stage('Unit Test') {
      steps {
        container('main') {
          sh 'echo Unit Test'
          // sh '/bin/bash --login -c "cd build && ctest -V -R system_allocator_test"'
        }
      }
    }
    stage('Integration Test') {
      steps {
        container('main') {
          sh 'echo Integration Test'
        }
      }
    }
    stage('Daily Release') {
      // Test commit update
      agent {
        kubernetes {
          yamlFile 'ci/templates/musa.yaml'
          defaultContainer "main"
        }
      }
      when {
        beforeAgent true
        allOf {
          branch 'main'
          expression { ifTriggeredByTimer() }
        }
      }
      steps {
        container('main') {
          sh 'echo Daily Release in main'
          
        }
        container('release') {
          sh 'echo Daily Release in release'
        }
      }
    }
  }
}
