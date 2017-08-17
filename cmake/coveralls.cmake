# CMake script for code coverage.
# If _COVERALLS_UPLOAD is ON, it will upload json files to overalls.io automatically.

# Param _GCOV_EXECUTABLE        Gcov executable.
# Param _COVERAGE_SRCS          A list of coverage source files.
# Param _GIT_PR_ID              Git pull request number.
# Param _JSON_GIT_INFO          Json format of git information  
# Param _COVERALLS_UPLOAD       Upload the result to coveralls.
# Param _CMAKE_SCRIPT_PATH      CMake script path.
function(code_coverage _GCOV_EXECUTABLE _COVERAGE_SRCS _GIT_PR_ID _JSON_GIT_INFO _COVERALLS_UPLOAD _CMAKE_SCRIPT_PATH)
    # clean previous gcov data.
    file(REMOVE_RECURSE ${PROJECT_BINARY_DIR}/*.gcda)

    # find curl for upload JSON soon.
    if (_COVERALLS_UPLOAD)
        find_program(CURL_EXECUTABLE curl)
        if (NOT CURL_EXECUTABLE)
            message(FATAL_ERROR "Coveralls: curl not found!")
        endif()
    endif()

    # When passing a CMake list to an external process, the list
    # will be converted from the format "1;2;3" to "1 2 3".
    set(COVERAGE_SRCS "")
    foreach (SINGLE_SRC ${_COVERAGE_SRCS})
        set(COVERAGE_SRCS "${COVERAGE_SRCS}*${SINGLE_SRC}")
    endforeach()

    # coveralls json file.
    set(COVERALLS_FILE ${PROJECT_BINARY_DIR}/coveralls.json)
    add_custom_target(coveralls_generate
        # Run unit tests.
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
        # Generate Gcov and translate it into coveralls JSON.
        COMMAND ${CMAKE_COMMAND}
                -DCOVERAGE_SRCS="${COVERAGE_SRCS}"
                -DCOVERALLS_OUTPUT_FILE="${COVERALLS_FILE}"
                -DCOV_PATH="${PROJECT_BINARY_DIR}"
                -DPROJECT_ROOT="${PROJECT_SOURCE_DIR}"
                -DGCOV_EXECUTABLE="${_GCOV_EXECUTABLE}"
                -DGIT_PR_ID="${_GIT_PR_ID}"
                -DJSON_GIT_INFO="${_JSON_GIT_INFO}"
                -P "${_CMAKE_SCRIPT_PATH}/coverallsGcovJsons.cmake"
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
        COMMENT "Coveralls: generating coveralls output..."
    )

    if (_COVERALLS_UPLOAD)
        # Upload the JSON to coveralls.
        add_custom_target(coveralls_upload
            COMMAND ${CURL_EXECUTABLE}
                    -S -F json_file=@${COVERALLS_FILE}
                    https://coveralls.io/api/v1/jobs
            DEPENDS coveralls_generate
            WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
            COMMENT "Coveralls: uploading coveralls output...")

        add_custom_target(coveralls DEPENDS coveralls_upload)
    else()
        add_custom_target(coveralls DEPENDS coveralls_generate)
    endif()
endfunction()

find_package(Git)

if (GIT_FOUND)
	# Branch.
	execute_process(
		COMMAND ${GIT_EXECUTABLE} rev-parse --abbrev-ref HEAD
		WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
		OUTPUT_VARIABLE GIT_BRANCH
		OUTPUT_STRIP_TRAILING_WHITESPACE
	)

	macro (git_log_format FORMAT_CHARS VAR_NAME)
		execute_process(
			COMMAND ${GIT_EXECUTABLE} log -1 --pretty=format:%${FORMAT_CHARS}
			WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
			OUTPUT_VARIABLE ${VAR_NAME}
			OUTPUT_STRIP_TRAILING_WHITESPACE
		)
	endmacro()

	git_log_format(an GIT_AUTHOR_NAME)
	git_log_format(ae GIT_AUTHOR_EMAIL)
	git_log_format(cn GIT_COMMITTER_NAME)
	git_log_format(ce GIT_COMMITTER_EMAIL)
	git_log_format(B GIT_COMMIT_MESSAGE)
	git_log_format(H GIT_COMMIT_HASH)
	git_log_format(ai GIT_DATE_ISO_8601)

	message("-- Git exe: ${GIT_EXECUTABLE}")
	message("-- Git branch: ${GIT_BRANCH}")
	message("-- Git author: ${GIT_AUTHOR_NAME}")
	message("-- Git author date: ${GIT_DATE_ISO_8601}")
	message("-- Git e-mail: ${GIT_AUTHOR_EMAIL}")
	message("-- Git commiter name: ${GIT_COMMITTER_NAME}")
	message("-- Git commiter e-mail: ${GIT_COMMITTER_EMAIL}")
	message("-- Git commit message: ${GIT_COMMIT_MESSAGE}")
	message("-- Git commit hash: ${GIT_COMMIT_HASH}")

	#
	# Store git commit infomation into coveralls json
	#
	# For example:
	#	"git": {
	#		"head": {
	#		  "id": "b31f08d07ae564b08237e5a336e478b24ccc4a65",
	#		  "author_name": "Nick Merwin",
	#		  "author_email": "...",
	#		  "committer_name": "Nick Merwin",
	#		  "committer_email": "...",
	#		  "message": "version bump"
	#		},
	#		"branch": "master",
	#		"remotes": [
	#		  {
	#			"name": "origin",
	#			"url": "git@github.com:lemurheavy/coveralls-ruby.git"
	#		  }
	#		]
	#	  },
	#

	set(JSON_GIT_INFO
	"{
	  \"head\": {
	    \"author_name\": \"${GIT_AUTHOR_NAME}\",
	    \"author_email\": \"${GIT_AUTHOR_EMAIL}\",
	    \"committer_name\": \"${GIT_COMMITTER_NAME}\",
	    \"committer_email\": \"${GIT_COMMITTER_EMAIL}\",
	    \"message\": \"${GIT_COMMIT_MESSAGE}\"
	  },
	  \"branch\": \"${GIT_BRANCH}\",
	  \"remotes\": [{
	    \"name\": \"origin\",
	    \"url\": \"https://github.com/PaddlePaddle/Paddle.git\"
	  }]
	}")
endif()

if(WITH_COVERAGE)
    set(CMAKE_BUILD_TYPE "Debug")
    set(CODE_COVERAGE_FLAGS "-g -O0 -fprofile-arcs -ftest-coverage")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CODE_COVERAGE_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CODE_COVERAGE_FLAGS}")

    file(GLOB_RECURSE PADDLE_SOURCES
        ${PADDLE_SOURCE_DIR}/paddle/memory/*.cc
        ${PADDLE_SOURCE_DIR}/paddle/memory/*.cu
        ${PADDLE_SOURCE_DIR}/paddle/memory/*.h
        ${PADDLE_SOURCE_DIR}/paddle/framework/*.h
        ${PADDLE_SOURCE_DIR}/paddle/framework/*.cu
        ${PADDLE_SOURCE_DIR}/paddle/framework/*.cc
        ${PADDLE_SOURCE_DIR}/paddle/operators/*.h
        ${PADDLE_SOURCE_DIR}/paddle/operators/*.cu
        ${PADDLE_SOURCE_DIR}/paddle/operators/*.cc
        ${PADDLE_SOURCE_DIR}/paddle/platform/*.h
        ${PADDLE_SOURCE_DIR}/paddle/platform/*.cu
        ${PADDLE_SOURCE_DIR}/paddle/platform/*.cc)
 
    # exclude trivial tests in PADDLE_SOURCES
    foreach(TEST_SOURCE ${PADDLE_SOURCES}) 
        string(REGEX MATCH "[/A-Za-z0-9_]*test.[a-z]*" TEST_SOURCE ${TEST_SOURCE})
        if (TEST_SOURCE)
            list(REMOVE_ITEM PADDLE_SOURCES ${TEST_SOURCE}) 
        endif()
    endforeach()

    message("-- Code Coverage: ${WITH_COVERAGE}")
    message("-- Coveralls Upload: ${COVERALLS_UPLOAD}")

    get_filename_component(CXX_DIR ${CMAKE_CXX_COMPILER} DIRECTORY)
    set(GCOV_EXECUTABLE ${CXX_DIR}/gcov)
    if (NOT GCOV_EXECUTABLE)
        message(FATAL_ERROR "gcov not found! Aborting...")
    else()
        message("-- Found gcov: ${GCOV_EXECUTABLE}")
    endif()

    # CI needs to pass pull request number into code coverage
    # in order to bind coveralls with github.
    set(GIT_PR_ID)
    code_coverage(
        "${GCOV_EXECUTABLE}"
        "${PADDLE_SOURCES}"
        "${GIT_PR_ID}"
        "${JSON_GIT_INFO}"
        ${COVERALLS_UPLOAD}
        "${PROJECT_SOURCE_DIR}/cmake"
    )
endif()
