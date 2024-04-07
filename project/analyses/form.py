import sys, os

sys.path.append('/Users/ilanashapiro/Documents/msaf0_1_80') # https://github.com/urinieto/msaf/tree/main
# sys.path.append('/Users/ilanashapiro/Library/musicaiz-0.1.2') # https://github.com/carlosholivan/musicaiz

import msaf
from concurrent.futures import ThreadPoolExecutor

def segment_audio_MSAF_test():
	audio_filepath = "LOP_database_06_09_17/liszt_classical_archives/1_short_test/beet_3_2_solo_short.mp3"

	# hierarchical: scluster, olda (boundaries only, use with fmc2d), vmo
	# sf (flat, boundaries only) has best performance
	boundaries, labels = msaf.process(audio_filepath, boundaries_id="scluster", labels_id="scluster", hier=True)
	out_file = audio_filepath[:-4] + '_segments.txt'
	
	print('Saving output to %s' % out_file)
	msaf.io.write_mirex_hierarchical(boundaries, labels, out_file)

# segment_audio_MSAF_test()

def segment_audio():
	directory = "/Users/ilanashapiro/Documents/constraints_project/project/classical_piano_midi_db"
	futures = []

	def process_file(file_path):
		# hierarchical: scluster, olda (boundaries only, use with fmc2d), vmo
		# sf (flat, boundaries only) has best performance
		boundaries, labels = msaf.process(file_path, boundaries_id="scluster", labels_id="scluster", hier=True)
		out_file = file_path[:-4] + '_segments.txt'
		print('Saving output to %s' % out_file)
		msaf.io.write_mirex_hierarchical(boundaries, labels, out_file)
	
	with ThreadPoolExecutor() as executor:
		for root, _, files in os.walk(directory):
			for filename in files:
				if filename.endswith(".mp3"):
					file_path = os.path.join(root, filename)
					future = executor.submit(process_file, file_path)
					futures.append(future)
	
# print(msaf.get_all_boundary_algorithms())
# print(msaf.get_all_label_algorithms())