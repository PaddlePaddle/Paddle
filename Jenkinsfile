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
          sh 'git diff --name-only origin/develop..HEAD | xargs pre-commit run --files'
        }
      }
    }
    stage('Build') {
      steps {
        container('main') {
          sh 'echo Build'
        }
      }
    }
    stage('Unit Test') {
      steps {
        container('main') {
          sh 'echo Unit Test'
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
