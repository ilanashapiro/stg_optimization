import json

from networkx import degree

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
		return {'start': float(start), 'end': float(end), 'id': node_id, 'index': idx, 'label': node_label}
		
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
			new_label = new_id.split('N')[0] # update labels too, not just node id's
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
						node_id = f"P{pattern_num}O{occurrence_num}"
						node_label = node_id
						motif_layer.append({'start': float(start), 'end': float(end), 'id': node_id, 'label': node_label, 'features_dict': {'pattern_num': pattern_num}})
					occurrence_num += 1
					start, end = None, None # Reset start and end for the new occurrence
				else:
					time, _ = line.split(',', 1)
					if start is None:
						start = time  # First line of occurrence sets the start time
					end = time
			# Add the last occurrence in the chunk
			if start is not None and end is not None and not(below_start_bound(start) and below_start_bound(end) or above_end_bound(start) and above_end_bound(end)):
				node_id = f"P{pattern_num}O{occurrence_num}"
				node_label = node_id
				motif_layer.append({'start': float(start), 'end': float(end), 'id': node_id, 'label': node_label, 'features_dict': {'pattern_num': pattern_num}})
		pattern_num += 1
		
	# Sort by start time and add index based on the sort
	motif_layer = sorted(motif_layer, key=lambda x: x['start'])
	for idx, item in enumerate(motif_layer, start=1):
		item['id'] += f"N{idx}"
		item['index'] = idx

	return motif_layer

def parse_melody_file(piece_start_time, piece_end_time, file_path):
	with open(file_path, 'r') as file:
		lines = file.read().strip().split('\n') 
	melody_layer = []
	melody_started = False

	def below_start_bound(time):
		return float(time) < float(piece_start_time) 
	def above_end_bound(time):
		return float(time) > float(piece_end_time)

	node_idx = 1
	for line in lines:
		line = line.strip()
		parts = line.split(')",')
		time_tuple_str = parts[0].strip('"()')
		start, end = map(float, time_tuple_str.split(','))

		if below_start_bound(start) and below_start_bound(end) or above_end_bound(start) and above_end_bound(end):
			continue 
		# for the first melody note ending within valid piece start/end time, we want to make sure it starts at piece start time
		if below_start_bound(start) or (not melody_started and start != piece_start_time): 
			start = piece_start_time
		# for the last melody note starting within valid piece start/end time, we want to make sure it starts at piece start time
		# when end > piece time, this means there's no more valid intervals bc of how we parsed the melody file
		if above_end_bound(end): 
			end = piece_end_time

		interval = int(float(parts[1].strip()))
		node_label = f"M{interval}"
		node_id = node_label + f"N{node_idx}" 
		melody_layer.append({'start': float(start), 'end': float(end), 'id': node_id, 'label': node_label, 'index': node_idx, 'features_dict': {'abs_interval': abs(interval), 'interval_sign': '+' if interval > 0 else '-'}})
		melody_started = True
		node_idx += 1

	return melody_layer

def parse_harmony_file(piece_start_time, piece_end_time, file_path):
	keys_layer = []
	chords_layer = []

	def get_relative_key_num(current_key, new_key):
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
		current_index = key_indices_assignments[current_key.upper()]
		new_index = key_indices_assignments[new_key.upper()]
		relative_index = (new_index - current_index) % 12
		return relative_index
	def below_start_bound(time):
		return float(time) < float(piece_start_time) 
	def above_end_bound(time):
		return float(time) > float(piece_end_time)
	
	with open(file_path, 'r') as file:
		key_idx = 1
		lines = file.readlines()
		curr_roman = None
		curr_key = None
		node_index = 1
		
		# only go up until, but not including, the last line since we process in pairs
		for line_idx in range(len(lines) - 1):
			line = lines[line_idx]
			curr_harmony_dict = json.loads(line.strip())
			key = curr_harmony_dict['key']
			onset_seconds = float(curr_harmony_dict['onset_seconds'])
			degree1 = curr_harmony_dict['degree1']
			degree2 = curr_harmony_dict['degree2']
			chord_quality = curr_harmony_dict['quality']
			roman = curr_harmony_dict['roman_chord']
			# inversion = harmony_dict['inversion'] # for now, let's leave this out of the graph, since it doesn't indicate significant harmonic change
			
			next_harmony_dict = json.loads(lines[line_idx+1].strip())
			next_onset_seconds = float(next_harmony_dict['onset_seconds'])

			if below_start_bound(onset_seconds) and below_start_bound(next_onset_seconds) or above_end_bound(onset_seconds) and above_end_bound(next_onset_seconds):
				continue
			
			if onset_seconds != next_onset_seconds:
				if below_start_bound(onset_seconds):
					onset_seconds = piece_start_time
				if above_end_bound(next_onset_seconds):
					next_onset_seconds = piece_end_time
				
				# since we're not considering inversion, we see if chords are equal based on degrees 1 and 2, and quality
				if curr_roman != roman:
					curr_roman = roman
					chord_node_id = f"C{degree1},{degree2}Q{chord_quality}N{node_index}" # functional harmony chord {degree1}, {degree2} quality {quality} number {number}
					chord_node_label = roman
					chords_layer.append({'start': onset_seconds, 'end': next_onset_seconds, 'id': chord_node_id, 'label': chord_node_label, 'index': node_index, 'features_dict': {'degree1': degree1, 'degree2': degree2, 'quality': chord_quality}})
					node_index += 1
				else: # Extend the end time of the current chord since it's just another inversion 
					chords_layer[-1]['end'] = next_onset_seconds

				if curr_key != key:
					relative_key_num = 0 if not curr_key else get_relative_key_num(curr_key, key) # first key is set to 1 for standardization
					key_quality = "M" if key[0].isupper() else "m"
					key_node_label = f"K{relative_key_num}Q{key_quality}"
					key_node_id = key_node_label + f"N{key_idx}" # functional harmony: key {relative_key_num} quality {quality} number {number}
					keys_layer.append({'start': onset_seconds, 'end': next_onset_seconds, 'id': key_node_id, 'label': key_node_label, 'index': key_idx, 'features_dict': {'relative_key_num': relative_key_num, 'quality': key_quality}})
					key_idx += 1
					curr_key = key
				else: # Extend the end time of the current key
					keys_layer[-1]['end'] = next_onset_seconds
		
		# process the last harmony dict
		last_harmony_dict = json.loads(lines[-1].strip())
		last_key = last_harmony_dict['key']
		last_onset_seconds = float(last_harmony_dict['onset_seconds'])
		last_degree1 = last_harmony_dict['degree1']
		last_degree2 = last_harmony_dict['degree2']
		last_chord_quality = last_harmony_dict['quality']
		last_roman = last_harmony_dict['roman_chord']

		if last_onset_seconds < piece_end_time:
			if curr_roman != last_roman:
				chord_node_id = f"C{last_degree1},{last_degree2}Q{last_chord_quality}N{node_index}" # functional harmony chord {degree1}, {degree2} quality {quality} number {number}
				chord_node_label = last_roman
				chords_layer.append({'start': last_onset_seconds, 'end': piece_end_time, 'id': chord_node_id, 'label': chord_node_label, 'index': node_index, 'features_dict': {'degree1': last_degree1, 'degree2': last_degree2, 'quality': last_chord_quality}})
			else: # Extend the end time of the current chord since it's just another inversion 
				chords_layer[-1]['end'] = piece_end_time

			if curr_key != last_key:
				relative_key_num = 0 if not curr_key else get_relative_key_num(curr_key, last_key) # first key is set to 1 for standardization
				last_key_quality = "M" if last_key[0].isupper() else "m"
				key_node_label = f"K{relative_key_num}Q{last_key_quality}"
				key_node_id = key_node_label + f"N{key_idx}" # functional harmony: key {relative_key_num} quality {quality} number {number}
				keys_layer.append({'start': last_onset_seconds, 'end': piece_end_time, 'id': key_node_id, 'label': key_node_label, 'index': key_idx, 'features_dict': {'relative_key_num': relative_key_num, 'quality': last_key_quality}})
			else: # Extend the end time of the current key
				keys_layer[-1]['end'] = piece_end_time

	return [keys_layer, chords_layer]