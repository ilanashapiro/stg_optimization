import sys, os
import pandas as pd

# DIRECTORY = "/home/ubuntu/project"
# DIRECTORY = "/home/ilshapiro/project"
DIRECTORY = "/Users/ilanashapiro/Documents/constraints_project/project"

# sys.path.append('/home/ubuntu/msaf_new')
sys.path.append('/home/ilshapiro/msaf_new') # https://github.com/urinieto/msaf/tree/main
# sys.path.append('/Users/ilanashapiro/Library/musicaiz-0.1.2') # https://github.com/carlosholivan/musicaiz

import msaf
from multiprocessing import Pool

def segment_audio_MSAF_test():
	audio_filepath = f"{DIRECTORY}/datasets/bach/classical_piano_midi_db/bach_850/bach_850.mp3"
	# hierarchical: scluster, olda (boundaries only, use with fmc2d), vmo
	# sf (flat, boundaries only) has best performance
	boundaries, labels = msaf.process(audio_filepath, boundaries_id="scluster", labels_id="scluster", hier=True)
	out_file = audio_filepath[:-4] + '_segments.txt'
	
	print('Saving output to %s' % out_file)
	msaf.io.write_mirex_hierarchical(boundaries, labels, out_file)
# segment_audio_MSAF_test()

def process_file(file_path):
	try:
		# hierarchical: scluster, olda (boundaries only, use with fmc2d), vmo
		# sf (flat, boundaries only) has best performance
		boundaries_algorithm, labels_algorithm, hierarchical = "scluster", "scluster", True
		out_file = file_path[:-4] + f"_{boundaries_algorithm}_{labels_algorithm}_segments.txt"
		if os.path.exists(out_file):
			return
		elif not (os.path.exists(file_path[:-4] + "_motives3.txt") or os.path.exists(file_path[:-4] + "_motives1.txt")):
			return
		print('Saving output to %s' % out_file)
		boundaries, labels = msaf.process(file_path, boundaries_id=boundaries_algorithm, labels_id=labels_algorithm, hier=hierarchical)
		if hierarchical:
			msaf.io.write_mirex_hierarchical(boundaries, labels, out_file)
		else:
			msaf.io.write_mirex(boundaries, labels, out_file)
	except Exception as e:
		print(f"Error processing {file_path}: {e}")


def segment_audio():
	directory = f"{DIRECTORY}/datasets"
	file_paths = []
	for root, _, files in os.walk(directory):
		for filename in files:
			if filename.endswith(".mp3"):
				file_path = os.path.join(root, filename)
				file_paths.append(file_path)
	
	with Pool() as pool:
		pool.map(process_file, file_paths)
	
if __name__ == '__main__':
	segment_audio()

# print(msaf.get_all_boundary_algorithms())
# print(msaf.get_all_label_algorithms())