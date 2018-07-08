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
    cmd = 'curl -vX POST http://localhost:5000/api -d @data_copy/{} --header "Content-Type: application/json"'.format(
        name)
    print(cmd)
    os.system(cmd)
