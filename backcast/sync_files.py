import os

files = os.listdir('data_output')
for f in files:
	output_file = 'data_output/' + f
	output_file_size = os.path.getsize(output_file)
	if output_file_size < 1000:
		print(f, output_file_size)
		backcast_file = 'data_backcast/' + f.replace('_output_', '_input_')
		os.remove(backcast_file)
		os.remove(output_file)
