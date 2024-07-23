import json

def parse_segments_file(file_path, piece_start_time, piece_end_time):
	with open(file_path, 'r') as file:
		data = file.read().strip().split('\n\n')  # Split into chunks by blank line

	def below_start_bound(time):
		return float(time) < float(piece_start_time) 
	def above_end_bound(time):
		return float(time) > float(piece_end_time)
	def create_label(start, end, idx):
		node_label = f"S{id}L{layer_idx + 1}"
		node_id = f"{node_label}N{idx}"
		return {'start': float(start), 'end': float(end), 'id': node_id, 'index': idx, 'label': node_id}
		
	segment_layers = []
	for layer_idx, chunk in enumerate(data):
		lines = chunk.split('\n')
		layer = []
		idx = 1
		for line in lines:
			start, end, id = line.split('\t')

			if below_start_bound(start) and below_start_bound(end) or above_end_bound(start) and above_end_bound(end):
				continue
			if below_start_bound(start):
				start = piece_start_time
			if above_end_bound(end):
				end = piece_end_time

			layer.append(create_label(start, end, idx))
			idx += 1
		segment_layers.append(layer)
	
	# fix section labels so each new label encountered is increasing from the previous
	for idx, layer in enumerate(segment_layers):
		s_num_mapping = {}
		new_s_num_counter = 0
		for node in layer:
			s_num = node['id'].split('L')[0][1:]  # This splits the id at 'L', takes the 'S{n1}' part, and then removes 'S' to get 'n1'
			if s_num not in s_num_mapping:
				s_num_mapping[s_num] = new_s_num_counter
				new_s_num_counter += 1

		updated_nodes = []
		for node in layer:
			old_s_num = node['id'].split('L')[0][1:]
			new_s_num = s_num_mapping[old_s_num]
			parts = node['id'].split('L') # this is e.g. "10N9"
			new_id = f'S{new_s_num}L{parts[1]}'
			new_label = new_id # update labels too, not just node id's
			updated_nodes.append({'start': node['start'], 'end': node['end'], 'index': node['index'], 'features_dict': {'section_num': new_s_num}, 'id': new_id, 'label': new_label})
		segment_layers[idx] = updated_nodes
	
	return segment_layers

def parse_motives_file(piece_start_time, piece_end_time, file_path):
	with open(file_path, 'r') as file:
		data = file.read().strip().split('\n\n')  # Split into chunks by blank line

	motif_layer = []
	pattern_num = 0

	def below_start_bound(time):
		return float(time) < float(piece_start_time) 
	def above_end_bound(time):
		return float(time) > float(piece_end_time)
	
	for chunk in data:
		if chunk.startswith("pattern"):
			pattern_num += 1
			lines = chunk.split('\n')[1:]  # Skip the pattern line itself
			occurrence_num = 0
			start, end = None, None
			for line in lines:
				if line.startswith("occurrence"):
					if start is not None and end is not None:
						if below_start_bound(start) and below_start_bound(end) or above_end_bound(start) and above_end_bound(end):
							continue
						if below_start_bound(start):
							start = piece_start_time
						if above_end_bound(end):
							end = piece_end_time
						# Save the previous occurrence before starting a new one
						node_label = f"P{pattern_num}O{occurrence_num}"
						motif_layer.append({'start': float(start), 'end': float(end), 'id': node_label, 'label': node_label, 'features_dict': {'pattern_num': pattern_num}})
					occurrence_num += 1
					start, end = None, None # Reset start and end for the new occurrence
				else:
					time, _ = line.split(',', 1)
					if start is None:
						start = time  # First line of occurrence sets the start time
					end = time
			# Add the last occurrence in the chunk
			if start is not None and end is not None and not(below_start_bound(start) and below_start_bound(end) or above_end_bound(start) and above_end_bound(end)):
				node_label = f"P{pattern_num}O{occurrence_num}"
				motif_layer.append({'start': float(start), 'end': float(end), 'id': node_label, 'label': node_label, 'features_dict': {'pattern_num': pattern_num}})

	# Sort by start time and add index based on the sort
	motif_layer = sorted(motif_layer, key=lambda x: x['start'])
	for idx, item in enumerate(motif_layer, start=1):
		item['id'] += f"N{idx}"
		item['label'] += f"N{idx}"
		item['index'] = idx

	return motif_layer

def parse_melody_file(piece_start_time, piece_end_time, file_path):
	with open(file_path, 'r') as file:
		data = file.read().strip().split('\n') 
	melody_layer = []

	for idx, line in enumerate(data):
		line = line.strip()
		parts = line.split(')",')
		time_tuple_str = parts[0].strip('"()')
		start, end = map(float, time_tuple_str.split(','))

		if start < piece_start_time and end < piece_start_time or start > piece_end_time and end > piece_end_time:
			continue 
		if start < piece_start_time:
			start = piece_start_time
		if end > piece_end_time:
			end = piece_end_time

		interval = int(float(parts[1].strip()))
		node_label = f"M{interval}N{idx}"
		melody_layer.append({'start': float(start), 'end': float(end), 'id': node_label, 'label': node_label, 'index': idx, 'features_dict': {'abs_interval': abs(interval), 'interval_sign': '+' if interval > 0 else '-'}})

	return melody_layer

def parse_harmony_file(piece_start_time, piece_end_time, file_path):
	keys_layer = []
	chords_layer = []
	
	key_indices_assignments = {
		'C': 1, 'B+': 1,
		'C+': 2, 'D-': 2,
		'D': 3,
		'D+': 4, 'E-': 4,
		'E': 5, 'F-': 5,
		'E+': 6, 'F': 6,
		'F+': 7, 'G-': 7,
		'G': 8,
		'G+': 9, 'A-': 9,
		'A': 10,
		'A+': 11, 'B-': 11,
		'B': 12
	}

	def get_relative_key_num(current_key, new_key):
		current_index = key_indices_assignments[current_key.upper()]
		new_index = key_indices_assignments[new_key.upper()]
		
		relative_index = (new_index - current_index) % 12
		return relative_index
	
	with open(file_path, 'r') as file:
		prev_key = None
		key_idx = 1
		lines = file.readlines()
		key_start_time = json.loads(lines[1].strip())['onset_seconds']

		# first line is the piece end time, only go up to the last line since we process in pairs
		for idx, line in enumerate(lines[1:-1], start=1):
			curr_harmony_dict = json.loads(line.strip())
			key = curr_harmony_dict['key']
			onset_seconds = float(curr_harmony_dict['onset_seconds'])
			degree1 = curr_harmony_dict['degree1']
			degree2 = curr_harmony_dict['degree2']
			quality = curr_harmony_dict['quality']
			# inversion = harmony_dict['inversion'] # for now, let's leave this out of the graph, since it doesn't indicate significant harmonic change

			next_harmony_dict = json.loads(lines[idx+1].strip())
			next_onset_seconds = float(next_harmony_dict['onset_seconds'])
			next_key = next_harmony_dict['key']

			if onset_seconds != next_onset_seconds and not(onset_seconds < piece_start_time and next_onset_seconds < piece_start_time or onset_seconds > piece_end_time and next_onset_seconds > piece_end_time):
				if onset_seconds < piece_start_time:
					onset_seconds = piece_start_time
				if next_onset_seconds > piece_end_time:
					next_onset_seconds = piece_end_time
				
				node_label = f"C{degree1},{degree2}Q{quality}N{idx}" # functional harmony chord {degree1}, {degree2} quality {quality} number {number}
				chords_layer.append({'start': onset_seconds, 'end': next_onset_seconds, 'id': node_label, 'label': node_label, 'index': idx, 'features_dict': {'degree1': degree1, 'degree2': degree2, 'quality': quality}})

				if key != next_key and next_onset_seconds > piece_start_time: # we have encountered a new key
					if key_start_time < piece_start_time:
						key_start_time = piece_start_time
					relative_key_num = 1 if not prev_key else get_relative_key_num(prev_key, key) # first key is set to 1 for standardization
					quality = "M" if key.isupper() else "m"
					node_label = f"K{relative_key_num}Q{quality}N{key_idx}" # functional harmony key {key} number {number}
					keys_layer.append({'start': key_start_time, 'end': next_onset_seconds, 'id': node_label, 'label': node_label, 'index': key_idx, 'features_dict': {'relative_key_num': relative_key_num, 'quality': quality}})
					key_idx += 1
					prev_key = key
					key_start_time = next_onset_seconds
		
		# process the last harmony dict
		last_harmony_dict = json.loads(lines[-1].strip())
		last_key = last_harmony_dict['key']
		last_onset_seconds = float(last_harmony_dict['onset_seconds'])
		degree1 = last_harmony_dict['degree1']
		degree2 = last_harmony_dict['degree2']
		quality = last_harmony_dict['quality']
		
		if last_onset_seconds < piece_end_time:
			node_label = f"C{degree1},{degree2}Q{quality}N{len(lines) - 1}" # functional harmony chord {degree1}, {degree2} quality {quality} number {number}
			chords_layer.append({'start': last_onset_seconds, 'end': piece_end_time, 'id': node_label, 'label': node_label, 'index': idx, 'features_dict': {'degree1': degree1, 'degree2': degree2, 'quality': quality}})
		
		if key_start_time < piece_end_time:
			if key_start_time < piece_start_time:
				key_start_time = piece_start_time
			relative_key_num = 1 if not prev_key else get_relative_key_num(prev_key, last_key) # first key is set to 1 for standardization
			quality = "M" if key.isupper() else "m"
			key_node_label = f"K{relative_key_num}Q{quality}N{key_idx}" # functional harmony key {key} number {number}
			keys_layer.append({'start': key_start_time, 'end': piece_end_time, 'id': key_node_label, 'label': key_node_label, 'index': key_idx, 'features_dict': {'relative_key_num': relative_key_num, 'quality': quality}})

	return [keys_layer, chords_layer]