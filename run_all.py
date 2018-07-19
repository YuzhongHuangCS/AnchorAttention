import os

prefix = 'data/'
files = os.listdir(prefix)

for i in range(len(files)):
	name = files[i]
	logfile = 'log/{}'.format(name.replace('.json', '.txt'))
	if not os.path.isfile(logfile):
		cmd = 'python3 main.py data/{} > {} 2>&1'.format(name, logfile)
		print(cmd)
		os.system(cmd)
