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
    cron(env.BRANCH_NAME == 'develop' ? '0 18 * * *' : '')
  }

  environment {
    paddle_musa_working_dir = '/paddle_musa'
  }

  stages {
    stage("CI-Clone") {
      // Move PR contents to the specific directory, mainly to make use of the pre-built artifacts(saving time for compiling)
      steps {
        container('main') {
          sh """#!/bin/bash
          mkdir -p ${env.paddle_musa_working_dir}
          cp -r ${env.WORKSPACE}/. ${env.paddle_musa_working_dir}
          """
        }
      }
    }
    stage("Codestyle-Check") {
      steps {
        // when using `dir()` directive, always raise java.nio.file.AccessDeniedException,
        // so change working directory within the shell scripts
        container('main') {
            sh """#!/bin/bash
            cd ${env.paddle_musa_working_dir}
            git config --global --add safe.directory "*"
            BRANCH=origin/develop /bin/bash tools/codestyle/pre_commit.sh
            #git diff --name-only origin/develop..HEAD | xargs pre-commit run --files
            """
        }
      }
    }
    stage('Build') {
      steps {
        container('main') {
          sh """#!/bin/bash
          cd ${env.paddle_musa_working_dir}
          /bin/bash ci/build.sh -j64
          """
          // sh 'cd ${env.paddle_musa_working_dir} && \
          //     if [ "BRANCH_NAME" = "develop" ]; then ref_revision=$($(git log -2 --pretty=%H | tail -1)); else ref_revision=origin/develop; fi; && \
          //     git diff --name-only ${ref_revision} | xargs touch && \
          //     find build -name "CMakeCache*" | xargs rm'
          // sh '/bin/bash --login -c "cp -r $PADDLE_MUSA_REPO_PATH/third_party/. third_party \
          //     && cp -r $PADDLE_MUSA_REPO_PATH/.git/modules .git"'
          // sh '/bin/bash --login -c "ls -l && ls -l paddle"'
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
          sh """#!/bin/bash
          cd ${env.paddle_musa_working_dir}/build
          ctest -V -R system_allocator_test
          """
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
      agent {
        kubernetes {
          yamlFile 'ci/templates/musa.yaml'
          defaultContainer "main"
        }
      }
      when {
        beforeAgent true
        allOf {
          branch 'develop'
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
