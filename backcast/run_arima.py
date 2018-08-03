import requests
import json
import sys
import os.path
import pdb

filename = sys.argv[1]
_cvt = json.loads(open(filename).read())
andy_response = requests.post("http://sage-rct.isi.edu:6002/forecast?quick=False", json=_cvt)
result = json.loads(andy_response.text)
output_filename = filename.replace('_input_', '_output_')
print('Done', output_filename)
with open(output_filename, 'w') as fout:
    json.dump(result, fout)
