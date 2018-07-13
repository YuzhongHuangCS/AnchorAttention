pip3 install -r requirements.txt
sed -i "s/backend      : TkAgg/backend      : Agg/" /usr/local/lib/python3.6/dist-packages/matplotlib/mpl-data/matplotlibrc
mkdir data
mkdir fig
mkdir output
mkdir model
mkdir log
mkdir upload
python3 app.py
