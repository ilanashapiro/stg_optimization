import sys, os
import pandas as pd

# sys.path.append('/home/jonsuss/Ilana_Shapiro/msaf_new')
# sys.path.append('/home/ubuntu/msaf_new')
sys.path.append('/Users/ilanashapiro/Documents/msaf_new') # https://github.com/urinieto/msaf/tree/main
# sys.path.append('/Users/ilanashapiro/Library/musicaiz-0.1.2') # https://github.com/carlosholivan/musicaiz

import msaf
from multiprocessing import Pool

def segment_audio_MSAF_test():
	# audio_filepath = "/Users/ilanashapiro/Documents/constraints_project/project/LOP_database_06_09_17/liszt_classical_archives/1_short_test/beet_3_2_solo_short.mp3"
	audio_filepath = "/Users/ilanashapiro/Documents/constraints_project/project/classical_piano_midi_db/bach/bach_850/bach_850.mp3"
	# audio_filepath = "/home/ubuntu/project/classical_piano_midi_db/bach/bach_850/bach_850.mp3"
	# audio_filepath = '/home/jonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/albeniz/alb_esp1/alb_esp1.mp3'
	# hierarchical: scluster, olda (boundaries only, use with fmc2d), vmo
	# sf (flat, boundaries only) has best performance
	boundaries, labels = msaf.process(audio_filepath, boundaries_id="scluster", labels_id="scluster", hier=True)
	out_file = audio_filepath[:-4] + '_segments.txt'
	
	print('Saving output to %s' % out_file)
	msaf.io.write_mirex_hierarchical(boundaries, labels, out_file)

# segment_audio_MSAF_test()

def process_file(file_path):
	try:
		print("PROCESSING", file_path)
		# hierarchical: scluster, olda (boundaries only, use with fmc2d), vmo
		# sf (flat, boundaries only) has best performance
		boundaries_algorithm, labels_algorithm = "scluster", "scluster"
		boundaries, labels = msaf.process(file_path, boundaries_id=boundaries_algorithm, labels_id=labels_algorithm, hier=True)
		out_file = file_path[:-4] + f"_{boundaries_algorithm}_{labels_algorithm}_segments.txt"
		
		print('Saving output to %s' % out_file)
		msaf.io.write_mirex_hierarchical(boundaries, labels, out_file)
	except Exception as e:
		print(f"Error processing {file_path}: {e}")


def segment_audio():
	directory = "/Users/ilanashapiro/Documents/constraints_project/project/classical_piano_midi_db/bach"
	# directory = '/home/jonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db'
	# directory = "/home/ubuntu/project/classical_piano_midi_db/bach"

	file_paths = []
	for root, _, files in os.walk(directory):
		for filename in files:
			if filename.endswith(".mp3"):
				file_path = os.path.join(root, filename)
				file_paths.append(file_path)
	
	# Create a pool of workers and distribute the file processing tasks
	with Pool() as pool:
		pool.map(process_file, file_paths)
	
if __name__ == '__main__':
	segment_audio()

# print(msaf.get_all_boundary_algorithms())
# print(msaf.get_all_label_algorithms())