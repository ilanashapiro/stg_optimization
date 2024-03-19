import sys

sys.path.append('/Users/ilanashapiro/Documents/msaf0_1_80') # https://github.com/urinieto/msaf/tree/main
# sys.path.append('/Users/ilanashapiro/Library/musicaiz-0.1.2') # https://github.com/carlosholivan/musicaiz

import msaf

def segment_audio_MSAF():
	audio_filepath = "LOP_database_06_09_17/liszt_classical_archives/1_short_test/beet_3_2_solo_short.mp3"

	# hierarchical: scluster, olda (boundaries only, use with fmc2d), vmo
	# sf (flat, boundaries only) has best performance
	boundaries, labels = msaf.process(audio_filepath, boundaries_id="scluster", labels_id="scluster", hier=True)
	out_file = audio_filepath[:-4] + '_segments.txt'
	
	print('Saving output to %s' % out_file)
	msaf.io.write_mirex_hierarchical(boundaries, labels, out_file)

# segment_audio_MSAF()
	
print(msaf.get_all_boundary_algorithms())
print(msaf.get_all_label_algorithms())