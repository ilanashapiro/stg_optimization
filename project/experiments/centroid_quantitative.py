import os, shutil, sys
import random
import json
import numpy as np
from multiprocessing import Pool

# dir_prefix = "/home/ilshapiro/project/"
dir_prefix = "/Users/ilanashapiro/Documents/constraints_project/project/"
# dir_prefix = "/home/jonsuss/Ilana_Shapiro/constraints/"
DIRECTORY = dir_prefix + "classical_piano_midi_db/"

sys.path.append(dir_prefix)
sys.path.append(f"{dir_prefix}/centroid")

import build_graph
import simanneal_centroid_helpers, simanneal_centroid, simanneal_centroid_run

def generate_augmented_graphs(composer_subdirs):
	augmented_STGs = []
	for subdir in composer_subdirs:
		subdir_path = os.path.join(DIRECTORY, subdir)
		if os.path.exists(subdir_path):
			for root, _, files in os.walk(subdir_path):
				segment_filepath = None
				motive_filepath = None
				for filename in files:
					full_path = os.path.join(root, filename)
					if 'motives4.txt' in filename and os.path.getsize(full_path) != 0:
						with open(full_path, 'r') as file:
							occurrence_count = file.read().count("occurrence")
							if occurrence_count < 60:
								motive_filepath = full_path
					elif 'sf_fmc2d_segments.txt' in filename:
						segment_filepath = full_path

				if motive_filepath and segment_filepath:
					G, _ = build_graph.generate_graph(segment_filepath, motive_filepath)
					build_graph.augment_graph(G)
					augmented_STGs.append(G)
	return augmented_STGs

def analyze_composer_subdir(base_dir):
	count_not_meeting_condition = 0
	total_meeting_condition = 0
	total_occ = 0

	# Traverse through all directories and subdirectories
	for subdir, _, files in os.walk(base_dir):
		found_file = False
		for filename in files:
			if 'motives4.txt' in filename:
				full_path = os.path.join(subdir, filename)
				if os.path.getsize(full_path) != 0:
					found_file = True
					# Count occurrences of the string "occurrence" in the file
					with open(full_path, 'r') as file:
						content = file.read()
						occurrences = content.count('occurrence')
						if occurrences < 60:
							print(f"Subdirectory: {subdir}, 'occurrence' count: {occurrences}")
							total_meeting_condition += 1
							total_occ += occurrences
					break  # No need to check other files in this subdir

		if not found_file or os.path.getsize(full_path) == 0:
			count_not_meeting_condition += 1

	# Print the number of subdirectories not meeting the condition
	print(f"Number of subdirectories without motives: {count_not_meeting_condition}")
	print(f"Number of subdirectories WITH motives: {total_meeting_condition}")
	print(f"Occurrence avg: {total_occ / total_meeting_condition}")
	return total_meeting_condition

def generate_centroid(composer_dir, composer):
	print("GENERATING CENTROID FOR", composer)
	STG_augmented_list = generate_augmented_graphs([composer_dir])
	listA_G, centroid_node_mapping = simanneal_centroid_helpers.pad_adj_matrices(STG_augmented_list)
	
	initial_alignment_files = [os.path.join(composer_dir, f'alignments/initial_alignment_{i}.txt') for i in range(len(listA_G))]
	initial_centroid_file = os.path.join(composer_dir, 'alignments/initial_centroid.txt')
	alignments_dir = os.path.join(composer_dir, 'alignments')
	
	if os.path.exists(alignments_dir):
		alignments = [np.loadtxt(f) for f in initial_alignment_files]
		initial_centroid = np.loadtxt(initial_centroid_file)
		print(f'Loaded existing centroid and alignment files from {composer_dir}')
	else:
		os.makedirs(alignments_dir)
		print(f"Created directory {alignments_dir}")

		# Random is faster
		# initial_centroid = random.choice(listA_G)
		# alignments = simanneal_centroid.get_alignments_to_centroid_parallel(initial_centroid, listA_G, centroid_node_mapping, 1.5, 0.01, 5000)
	
		# this WAY TOO LONG w/o cluster, it's n^2 annealings 
		initial_centroid, _, alignments = simanneal_centroid_run.initial_centroid_and_alignments(listA_G, centroid_node_mapping)
	
		np.savetxt(initial_centroid_file, initial_centroid)
		print(f'Saved: {initial_centroid_file}')
		
		for i, alignment in enumerate(alignments):
			file_name = initial_alignment_files[i]
			np.savetxt(file_name, alignment)
			print(f'Saved: {file_name}')

	aligned_listA_G = list(map(simanneal_centroid.align, alignments, listA_G))
	centroid_annealer = simanneal_centroid.CentroidAnnealer(initial_centroid, aligned_listA_G, centroid_node_mapping)
	centroid_annealer.Tmax = 2.5
	centroid_annealer.Tmin = 0.05 
	centroid_annealer.steps = 100
	centroid, min_loss = centroid_annealer.anneal()

	centroid, centroid_node_mapping = simanneal_centroid.helpers.remove_dummy_nodes(centroid, centroid_node_mapping)

	np.savetxt(os.path.join(composer_dir, f"centroid_{composer}.txt"), centroid)
	print(f'Saved: {os.path.join(composer_dir, f"centroid_{composer}.txt")}')
	with open(os.path.join(composer_dir, f"centroid_node_mapping_{composer}.txt"), 'w') as file:
		json.dump(centroid_node_mapping, file)
		print(f'Saved: {file}')
	print("Best centroid", centroid)
	print("Best loss", min_loss)

def delete_initial_alignments_dirs(composers):
	for composer in composers:
		alignments_dir = os.path.join(os.path.join(DIRECTORY, composer), 'alignments')
		if os.path.exists(alignments_dir):
			shutil.rmtree(alignments_dir)
			print(f"Deleted directory {alignments_dir}")
		else:
			print("Directory does not exist, no action taken.")

if __name__ == "__main__":
	analyze_composer_subdir(os.path.join(DIRECTORY, "haydn"))
	sys.exit(0)
	composers = ["albeniz", "bach", "schumann", "haydn"]
	# delete_initial_alignments_dirs(composers)
	for composer in composers:
		generate_centroid(os.path.join(DIRECTORY, composer), composer)