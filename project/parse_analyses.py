import json

from networkx import degree

def parse_form_file(file_path):
	with open(file_path, 'r') as file:
		data = file.read().strip().split('\n\n')  # Split into chunks by blank line

	segment_layers = []
	for layer_idx, chunk in enumerate(data):
		lines = chunk.split('\n')
		layer = []
		for idx, line in enumerate(lines):
			start, end, id = line.split('\t')
			node_label = f"S{id}L{layer_idx + 1}"
			node_id = f"{node_label}N{idx}"
			layer.append({'start': float(start), 'end': float(end), 'id': node_id, 'label': node_id})
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
			parts = node['id'].split('L')
			new_id = f'S{new_s_num}L{parts[1]}'
			new_label = new_id # update labels too, not just node id's
			updated_nodes.append({'start': node['start'], 'end': node['end'], 'id': new_id, 'label': new_label})
		segment_layers[idx] = updated_nodes

	return segment_layers

def parse_motives_file(file_path):
	with open(file_path, 'r') as file:
		data = file.read().strip().split('\n\n')  # Split into chunks by blank line

	motif_layer = []
	pattern_num = 0

	for chunk in data:
		if chunk.startswith("pattern"):
			pattern_num += 1
			lines = chunk.split('\n')[1:]  # Skip the pattern line itself
			occurrence_num = 0
			start, end = None, None
			for line in lines:
				if line.startswith("occurrence"):
					if start is not None and end is not None:
						# Save the previous occurrence before starting a new one
						node_label = f"P{pattern_num}O{occurrence_num}"
						motif_layer.append({'start': float(start), 'end': float(end), 'id': node_label, 'label': node_label})
					occurrence_num += 1
					start, end = None, None # Reset start and end for the new occurrence
				else:
					time, _ = line.split(',', 1)
					if start is None:
						start = time  # First line of occurrence sets the start time
					end = time
			# Add the last occurrence in the chunk
			if start is not None and end is not None:
				node_label = f"P{pattern_num}O{occurrence_num}"
				motif_layer.append({'start': float(start), 'end': float(end), 'id': node_label, 'label': node_label})

	# Sort by start time and add index based on the sort
	motif_layer = sorted(motif_layer, key=lambda x: x['start'])
	for idx, item in enumerate(motif_layer):
		item['id'] += f"N{idx}"
		item['label'] += f"N{idx}"

	return motif_layer

def parse_melody_file(file_path):
	with open(file_path, 'r') as file:
		data = file.read().strip().split('\n') 
	melody_layer = []

	for idx, line in enumerate(data):
		line = line.strip()
		parts = line.split(')",')
		time_tuple_str = parts[0].strip('"()')
		start, end = map(float, time_tuple_str.split(','))
		label = int(float(parts[1].strip()))
		node_label = f"M{label}N{idx}"
		melody_layer.append({'start': float(start), 'end': float(end), 'id': node_label, 'label': node_label})

	return melody_layer

def parse_harmony_file(file_path):
	key_layer = []
	fh_layer = []
	
	with open(file_path, 'r') as file:
		current_key = None
		key_start_time = None
		prev_line_start_time = None
		key_idx = 0
		line_idx = 0

		for line in file:
			harmony_dict = json.loads(line.strip())
			key = harmony_dict['key']
			onset_seconds = harmony_dict['onset_seconds']
			degree1 = harmony_dict['degree1']
			degree2 = harmony_dict['degree2']
			quality = harmony_dict['quality']
			# inversion = harmony_dict['inversion'] # for now, let's leave this out of the graph, since it doesn't indicate significant harmonic change
			
			if key_start_time and key != current_key:
				if not current_key:
					raise Exception("Current key is None in", file_path)
				node_label = f"K{key}N{key_idx}"
				key_layer.append({'start': float(key_start_time), 'end': float(onset_seconds), 'id': node_label, 'label': node_label})
				current_key = key
				key_start_time = onset_seconds
				key_idx += 1
			
			node_label = f"FH{degree1},{degree2}Q{quality}N{line_idx}"
			if prev_line_start_time:
				fh_layer.append({'start': float(prev_line_start_time), 'end': float(onset_seconds), 'id': node_label, 'label': node_label})
			prev_line_start_time = onset_seconds
			line_idx += 1

		return [key_layer, fh_layer]