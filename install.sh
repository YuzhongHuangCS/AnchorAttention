#./hyper run -d -p 80 -p 22 -p 5000 --name rnn --size m1 rastasheep/ubuntu-sshd:18.04

apt-get update
apt-get install -y htop python3-pip git
cd
git clone https://yuzhongh@bitbucket.org/yuzhongh/sage-rnn.git
cd sage-rnn
pip3 install -r requirements.txt
sed -i "s/backend      : TkAgg/backend      : Agg/" /usr/local/lib/python3.6/dist-packages/matplotlib/mpl-data/matplotlibrc
mkdir data
mkdir fig
mkdir output
mkdir model
mkdir log
python3 app.py