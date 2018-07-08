import os
import sys

prefix = 'data_copy/'
files = os.listdir(prefix)

if len(sys.argv) > 2:
    start = int(sys.argv[1])
    step = int(sys.argv[2])
else:
    start = 0
    step = 1

for i in range(start, len(files), step):
    name = files[i]
    cmd = 'python regular.py data_copy/{} > log/{} 2>&1'.format(name, name.replace('.json', '.log'))
    print(cmd)
    os.system(cmd)
