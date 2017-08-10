#!/bin/bash

ARGPARSE=`getopt -o u:vin:l:e: --long git-user:,help,dry-run,task-name:,log-file:,exit-code:  -- "$@"`
KEEP_ANONYMOUS="A_USER_DOES_NOT_TELL_US"
# paddle config home dir, same as paddle
PADDLE_CONF_HOME="$HOME/.config/paddle"
# api url, mirror url(s) will be append later
PD_URLS="http://api.paddlepaddle.org/version"

usage()
{
    echo "Usage: `basename $0` [options]"
    echo "Options:"
    echo "  -e, --exit-code=EXIT_CODE         The train/predict process's exit code"
    echo "  -l, --log-file=LOG_FILE_PATH      Read which log file to get the duration of process"
    echo "  -n, --task-name=TASK_NAME         The name of demo or example"
    echo "  -u, --git-user=GITHUB_USER        provide contact info, like username or email"
    echo "  -v, -i                            Verbose output and interact with user when necessary"
    echo " --help                             display this help message"
}

eval set -- "${ARGPARSE}"
while true; do
    case "$1" in
        -l|--log-file)
            log_file=$2
            shift 2
            ;;
        -e|--exit-code)
            exit_code=$2
            shift 2
            ;;
        -u|--git-user)
            github_user=$2
            shift 2
            ;;
        -n|--task-name)
            task=$2
            shift 2
            ;;
        -v|-i)
            v=1
            shift
            ;;
        --dry-run)
            dry_run=1
            shift
            ;;
        --)
            shift
            break
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Invalid option $1"
            usage
            exit 1
            ;;
    esac
done

# parse the log_file to get the time costs
if [ -s "${log_file}" ]; then
    duration=`awk 'BEGIN{day=0;last_sec=0;min_sec=0;max_sec=0;}
    {if(index($2,":")==3){
        t=substr($2,1,8);
        sec=day*86400+substr(t,1,2)*3600+substr(t,4,2)*60+substr(t,7,2);
        if(sec<last_sec-600){day+=1;sec+=86400;}
        last_sec=sec;
        if(min_sec==0 || min_sec>sec){min_sec=sec;}
        if(max_sec==0 || max_sec<sec){max_sec=sec;}
    }}
    END{print max_sec-min_sec}' ${log_file}`
else
    duration=-1
fi
if [ "${v}" = "1" ]; then echo "duration: ${duration}"; fi

# try find the user/email if not given
if [ -z "${github_user}" ]; then
    # search for cached username
    if [ -s "${PADDLE_CONF_HOME}/github_user" ]; then
        if [ "${v}" = "1" ]; then echo "read github_user from cache..."; fi
        github_user=`cat ${PADDLE_CONF_HOME}/github_user`
    else
        # search the github-user from git config
        if [ "${v}" = "1" ]; then echo "read github_user from git..."; fi
        git_username=`git config --get user.name 2>/dev/null`
        git_url=`git config --get remote.origin.url 2>/dev/null`
        if [ "`echo ${git_url} | cut -b 1-19`" = "https://github.com/" ]; then
            # under a git url, like https://github.com/user_xxx/proj_yyy.git
            if [ "${v}" = "1" ]; then echo " from github url..."; fi
            github_user=`echo ${git_url} | cut -d "/" -f 4`
            if [ "${github_user}" = "PaddlePaddle" ]; then
                github_user=
            fi
        fi
        if [ -n "${git_username}" -a -z "${github_user}" ]; then
            if [ "${v}" = "1" ]; then echo " from global git username..."; fi
            github_user=${git_username}
        fi
    fi
fi
# allow user to set the user name, if it's not found
if [ -z "${github_user}" -a "${v}" = "1" ]; then
    read -p "Please input your github username or email, or just return to keep this feedback anonymous:"
    github_user=${REPLY}
    if [ -z "${github_user}" ]; then
        # empty input, consider as one anonymous user
        github_user="${KEEP_ANONYMOUS}"
    fi
fi
if [ -n "${github_user}" -a -z "${dry_run}" ]; then
    # valid user and not in dry-run mode, then save to cache
    mkdir -p ${PADDLE_CONF_HOME}
    echo "${github_user}" >${PADDLE_CONF_HOME}/github_user
fi
if [ "${v}" = "1" ]; then echo "username: ${github_user}"; fi
if [ "${github_user}" = "${KEEP_ANONYMOUS}" ]; then
    # anonymous user should keep the var empty.
    github_user=
fi

# read local paddle version
paddle_version=`paddle version | grep PaddlePaddle | head -n1 | cut -d " " -f 2 | cut -d "," -f 1`
if [ "${v}" = "1" ]; then echo "version:${paddle_version}"; fi

# read local system time
system_time=`date "+%Y%m%d%H%M%S"`
if [ "${v}" = "1" ]; then echo "system time:${system_time}"; fi

# make empty job_name as default value.
if [ -z "${task}" ]; then
    task="(unknown_task)"
fi
if [ "${v}" = "1" ]; then echo "task: ${task}"; fi

# concat the curl command
params="content={\"data_type\":\"usage\",\
\"system_time\":${system_time},\"paddle_version\":\"${paddle_version}\",\
\"github_user\":\"${github_user}\",\"job_name\":\"${task}\",\
\"duration\":${duration},\"exit_code\":\"${exit_code}\"\
}&type=1"
curl_cmd_prefix="curl -m 5 -X POST -d ${params}\
 -b ${PADDLE_CONF_HOME}/paddle.cookie -c ${PADDLE_CONF_HOME}/paddle.cookie "

if [ "${dry_run}" = "1" ]; then
    first_url=`echo ${PD_URLS} | cut -d " " -f 1`
    echo "(dry-run mode)curl command: ${curl_cmd_prefix} ${first_url}"
    exit 0
else
    for u in ${PD_URLS}; do
        curl_cmd="${curl_cmd_prefix} ${u}"
        if [ "${v}" = "1" ]; then echo "run: ${curl_cmd}"; fi
        ${curl_cmd} >/dev/null 2>&1
        if [ $? -eq 0 ]; then
            if [ "${v}" = "1" ]; then echo "upload OK!"; fi
            exit 0
        else
            if [ "${v}" = "1" ]; then echo "upload failed...try next"; fi
        fi
    done
    if [ "${v}" = "1" ]; then echo "all urls tried but all failed...exit"; fi
    exit 1
fi
