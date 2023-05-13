# Install project dependencies, without streamlit
sudo pip3 install $(grep -v streamlit requirements.txt)

# Install streamlit dependencies, without pyarrow
sudo pip3 install 'altair<5,>=3.2.0' \
    'blinker>=1.0.0' \
    'cachetools>=4.0' \
    'click>=7.0' \
    'importlib-metadata>=1.4' \
    'numpy' \
    'packaging>=14.1' \
    'pandas<3,>=0.25' \
    'pillow>=6.2.0' \
    'protobuf>=3.20,<5' \
    'pympler>=0.9' \
    'python-dateutil' \
    'requests>=2.4' \
    'rich>=10.11.0' \
    'tenacity<9,>=8.0.0' \
    'toml' \
    'typing-extensions>=3.10.0.0' \
    'tzlocal>=1.1' \
    'validators>=0.2' \
    'watchdog' \
    'pympler>=0.9' \
    'tenacity<9,>=8.0.0' \
    'validators>=0.2' \
    'tornado'

# Install steamlit without deps
sudo pip3 install streamlit --no-dependencies

# Grab a fake pyarrow
[ -f mock_pyarrow.py ] || wget https://raw.githubusercontent.com/dorinclisu/camplayer_streamlit/main/src/mock_pyarrow.py

# Now run app.py (hopefully)
PYTHON_CODE=$(cat <<END

import mock_pyarrow

try: from streamlit.web.cli import main
except: from streamlit.cli import main

main(prog_name='streamlit')

END
)


sudo python3 -c "$PYTHON_CODE" "$@"
