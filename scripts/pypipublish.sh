#! /bin/zsh
# pypipublish.sh
# Usage: From root of lastmile_utils repo, run ./scripts/pypipublish.sh

# NOTE: This assumes you have the lastmile_utils conda environment created.
# You will be prompted for a username and password. For the username, use __token__. 
# For the password, use the token value, including the pypi- prefix.
# To get a PyPi token, go here: 
# Need to get a token from here (scroll down to API Tokens): https://pypi.org/manage/account/ 
# If you have issues, read the docs: https://packaging.python.org/en/latest/tutorials/packaging-projects/ 

cd python 
rm -rf ./dist
source /opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh && conda activate lastmile_utils
python3 -m pip install --upgrade build
python3 -m build
python3 -m pip install --upgrade twine

# If you want to upload to testpypi, run pypipublish-test.sh.

python3 -m twine upload dist/*

cd ..
