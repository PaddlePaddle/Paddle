set work_dir=%cd%
if not defined cache_dir set cache_dir=%work_dir:Paddle=cache%
if not exist %cache_dir%\tools (
    cd /d %cache_dir%
    python -m pip install wget
    python -c "import wget;wget.download('https://paddle-ci.gz.bcebos.com/window_requirement/tools.zip')"
    tar xf tools.zip
    cd /d %work_dir%
)

pip config set global.trusted-host pypi.org
pip config set global.trusted-host files.pythonhosted.org
pip config set global.trusted-host pypi.python.org
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set install.trusted-host pypi.tuna.tsinghua.edu.cn
git config --global core.longpaths true
git config --global user.name "PaddleCI"
git config --global user.email "paddle_ci@example.com"

git remote add upstream https://github.com/PaddlePaddle/Paddle.git

git --no-pager pull upstream %BRANCH% --no-edit
if %errorlevel% NEQ 0 exit /b 1
if exist .git\index.lock del .git\index.lock 2>NUL
