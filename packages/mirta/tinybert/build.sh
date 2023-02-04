set -e

virtualenv virtualenv
source virtualenv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt --target virtualenv/lib/python3.9/site-packages