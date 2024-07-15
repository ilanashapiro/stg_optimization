import numpy as np
import tensorflow as tf # version = 1.8.0
import time
import random
import math
import pickle
from collections import Counter, namedtuple
import chord_recognition_models as crm
import string 

# Disables AVX/FMA
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Mappings of functional harmony
'''key: 7 degrees * 3 accidentals * 2 modes + 1 padding= 43'''
key_dict = {}
for i_a, accidental in enumerate(['', '#', 'b']):
		for i_t, tonic in enumerate(['C', 'D', 'E', 'F', 'G', 'A', 'B', 'c', 'd', 'e', 'f', 'g', 'a', 'b']):
				key_dict[tonic + accidental] = i_a * 14 + i_t
				if accidental == '#':
						key_dict[tonic + '+'] = i_a * 14 + i_t
				elif accidental == 'b':
						key_dict[tonic + '-'] = i_a * 14 + i_t
key_dict['pad'] = 42

'''degree1: 10 (['1', '2', '3', '4', '5', '6', '7', '-2', '-7', 'pad'])'''
degree1_dict = {d1: i for i, d1 in enumerate(['1', '2', '3', '4', '5', '6', '7', '-2', '-7', 'pad'])}

'''degree2: 15 ['1', '2', '3', '4', '5', '6', '7', '+1', '+3', '+4', '-2', '-3', '-6', '-7', 'pad'])'''
degree2_dict = {d2: i for i, d2 in enumerate(['1', '2', '3', '4', '5', '6', '7', '+1', '+3', '+4', '-2', '-3', '-6', '-7', 'pad'])}

'''quality: 11 (['M', 'm', 'a', 'd', 'M7', 'm7', 'D7', 'd7', 'h7', 'a6', 'pad'])'''
quality_dict = {q: i for i, q in enumerate(['M', 'm', 'a', 'd', 'M7', 'm7', 'D7', 'd7', 'h7', 'a6', 'pad'])}
quality_dict['a7'] = [v for k, v in quality_dict.items() if k == 'a'][0]

def invert_dict(d):
	return {v: k for k, v in d.items()}

def translate_predictions(data, pred_cc_values, pred_k_values, pred_r_values, eval_mask_values):
	'''pred_cc_values:
	Each row corresponds to a sequence in the batch.
	Each column corresponds to a time step within that sequence.
	The values (0 or 1) indicate the model's prediction for whether a chord change occurs at that time step.
	
	pred_k_values:
	Each row corresponds to a sequence in the batch.
	Each column corresponds to a time step within that sequence.
	The values are integers representing the predicted key for each time step.
	
	pred_r_values:
	Each row corresponds to a sequence in the batch.
	Each column corresponds to a time step within that sequence.
	The values are integers representing the predicted Roman numeral, which can be decoded back to musical terms using degree1_dict, degree2_dict, quality_dict, and inversion.
	
	The eval_mask tensor is used to distinguish between valid and padding parts of the sequences during evaluation.
	same dimensions
	'''

	# In the reshaped pianoroll array stored in corpus_aug_reshape[shift_id][piece_name]['pianoroll'], the axes represent the following:
	# Axis 0: Represents the sequences or segments of the pianoroll.
	# Axis 1: Represents the time steps within each sequence.
	# Axis 2: Represents the MIDI pitches.

	# from original processing: 
	# time = range(onset, end)
	# pianoroll[pitch-21, time] = 1 # add note to representation
	# if time = range(onset, end), then pianoroll[pitch-21, time] = 1 effectively sets all the time steps 
	# within the range onset to end for a given pitch to 1, indicating that a note of that pitch is active 
	# during those time steps.

	#  Padding and Reshaping
	# After padding and reshaping, the pianoroll is segmented into smaller sequences:

	# Padding:
	# pianoroll_pad has shape [total_length + n_pad, 88].
	# Padding ensures that the length is divisible by n_steps.

	# Reshaping:
	# pianoroll_pad is reshaped into sequences of fixed length (n_steps).

	# Reshaped pianoroll Structure
	# The reshaped pianoroll stored in corpus_aug_reshape[shift_id][piece_name]['pianoroll'] has shape [-1, n_steps, 88].
	# Axis 0: Represents the sequences (segments) of the pianoroll.
	# Axis 1: Represents the time steps within each sequence.
	# Axis 2: Represents the MIDI pitches.
	

	[inv_key_dict, inv_degree1_dict, inv_degree2_dict, inv_quality_dict] = [invert_dict(d) for d in [key_dict, degree1_dict, degree2_dict, quality_dict]]

	results = []

	pianoroll = data['pianoroll'] # shape: (n_sequences, n_timesteps_per_padded_seq, n_pitches)
	piece_ids = data['piece_id']

	# The len array tells you the length of each UNPADDED sequence in the pianoroll array.
	# e.g. the first value in len corresponds to the length of the first sequence in pianoroll.
	lens = data['len']
	num_sequences = len(pred_cc_values)

	all_original_timesteps = []
	current_time = 0
	current_piece_id = piece_ids[0]  # Initialize with the first piece_id

	for seq_idx, seq_len in enumerate(lens):
		all_original_timesteps.append(np.arange(current_time, current_time + seq_len))
		current_time += seq_len
		
		# Check if we have reached the end of a piece
		if seq_idx < len(piece_ids) - 1 and piece_ids[seq_idx + 1] != current_piece_id:
			# Move to the start of the next piece
			current_piece_id = piece_ids[seq_idx + 1]
			current_time = 0  # Reset current_time for the new piece

	prev_chord = {}
	for seq_idx in range(num_sequences):
		valid_timesteps_mask = eval_mask_values[seq_idx]
		# valid_pred_cc = pred_cc_values[seq_idx][valid_timesteps_mask] # dimensions: [num_valid_timesteps]
		valid_pred_k = pred_k_values[seq_idx][valid_timesteps_mask] # dimensions: [num_valid_timesteps]
		valid_pred_r = pred_r_values[seq_idx][valid_timesteps_mask] # dimensions: [num_valid_timesteps]

		piece_id = piece_ids[seq_idx]
		original_timesteps = all_original_timesteps[seq_idx]

		for t_idx, t in enumerate(original_timesteps):
			# val_cc = valid_pred_cc[t_idx] # this is the chord change value predicted by the neural net. however, it may not be totally accurate, i.e. align with pred_k and pred_r changing
			# if val_cc: # if we encounter a chord change at this timestep
				val_r = valid_pred_r[t_idx]
				if val_r == 9 * 14 * 10 * 4:  # Padding
					print("ERROR IN", piece_id, "AT", t) # this should have been removed because eval_mask was already applied
					sys.exit(0)

				# test_data_label_roman = test_data_label_degree1 * 14 * 10 * 4 + test_data_label_degree2 * 10 * 4 + test_data_label_quality * 4 + test_data_label_inversion
				# test_data_label_roman[test_data['label']['key'] == 'pad'] = 9 * 14 * 10 * 4
				degree1 = val_r // (14 * 10 * 4)
				val_r %= (14 * 10 * 4)
				degree2 = val_r // (10 * 4)
				val_r %= (10 * 4)
				quality = val_r // 4
				inversion = val_r % 4

				val_k = inv_key_dict[valid_pred_k[t_idx]] # Since pred_cc is binary (0 or 1), no decoding needed
				
				def are_chords_equal(chord1_dict, chord2_dict):
					# List of keys to compare, excluding 'timestep'
					keys_to_compare = ['piece_id', 'key', 'degree1', 'degree2', 'quality', 'inversion']
					
					for key in keys_to_compare:
						if chord1_dict.get(key) != chord2_dict.get(key):
							return False
					return True

				curr_chord = {
					'timestep': t,
					'piece_id': piece_id,
					# 'chord_change': val_cc,
					'key': val_k,
					'degree1': degree1,
					'degree2': degree2,
					'quality': quality,
					'inversion': inversion
				}

				if not are_chords_equal(curr_chord, prev_chord): 
					# print("CURR CHORD", curr_chord)
					results.append(curr_chord)

				prev_chord = curr_chord

	return results

def load_data_unlabeled(filepath, chunk_size=20):
		print(f"Loading data in {filepath}...")
		with open(filepath, 'rb') as file:
				corpus_aug_reshape = pickle.load(file)
		
		number_of_pieces = len(corpus_aug_reshape['shift_0'].keys())
		print("Number of pieces", number_of_pieces)

		print("START TIME", corpus_aug_reshape['shift_0']['vierne_sinfonia_2_4_(c)borsari']['start_time'])
		sys.exit(0)

		pianorolls = []
		tonal_centroids = []
		lens = []
		piece_ids = []

		for piece_id in corpus_aug_reshape['shift_0'].keys():
			'''corpus_aug_reshape[shift_id][op]['key'][0]: non-overlaped sequences
												corpus_aug_reshape[shift_id][op]['key'][1]: overlapped sequences'''
			# After reshaping, the pianoroll in corpus_aug_reshape will have the shape [num_sequences, n_steps, 88]
			# Here, num_sequences is the total number of sequences after padding and reshaping.
			# n_steps is the fixed number of time steps per sequence (128 in your case).
			# 88 corresponds to the MIDI pitch range (if MIDI note values range from 0 to 87).
			pianorolls.append(corpus_aug_reshape['shift_0'][piece_id]['pianoroll'][0])
			tonal_centroids.append(corpus_aug_reshape['shift_0'][piece_id]['tonal_centroid'][0])
			sequence_lens = corpus_aug_reshape['shift_0'][piece_id]['len'][0]
			lens.append(sequence_lens)
			piece_ids.append([piece_id] * len(sequence_lens))
			

		# total_pieces = len(pianorolls)
		# chunked_data = []

		# for i in range(0, total_pieces, chunk_size):
		#   chunk = {
		#       'pianoroll': np.concatenate(pianorolls[i:i + chunk_size], axis=0),
		#       'tonal_centroid': np.concatenate(tonal_centroids[i:i + chunk_size], axis=0),
		#       'len': np.concatenate(lens[i:i + chunk_size], axis=0)
		#   }
		#   chunked_data.append(chunk)

		# print('Keys in corpus_aug_reshape[\'shift_0\'][\'piece_id\'] =', list(corpus_aug_reshape['shift_0'].values())[0].keys())
		# print('Keys in data =', chunked_data[0].keys())
		# return chunked_data

		data = {
				'pianoroll': np.concatenate(pianorolls, axis=0),
				'tonal_centroid': np.concatenate(tonal_centroids, axis=0),
				'len': np.concatenate(lens, axis=0),
				'piece_id': np.concatenate(piece_ids, axis=0)
		}

		print('Keys in corpus_aug_reshape[\'shift_id\'][\'piece_id\'] =', list(corpus_aug_reshape['shift_0'].values())[0].keys())
		print('Keys in data =', data.keys())
		return data

def load_data_functional(dir, test_set_id=1, sequence_with_overlap=True):
		if test_set_id not in [1, 2, 3, 4]:
				print('Invalid testing_set_id.')
				exit(1)

		print("Load functional harmony data ...")
		print('test_set_id =', test_set_id)
		with open(dir, 'rb') as file:
				corpus_aug_reshape = pickle.load(file)
		print('keys in corpus_aug_reshape[\'shift_id\'][\'op\'] =', corpus_aug_reshape['shift_0']['1'].keys())

		shift_list = sorted(corpus_aug_reshape.keys())
		number_of_pieces = len(corpus_aug_reshape['shift_0'].keys())
		train_op_list = [str(i + 1) for i in range(number_of_pieces) if i % 4 + 1 != test_set_id]
		test_op_list = [str(i + 1) for i in range(number_of_pieces) if i % 4 + 1 == test_set_id]
		print('shift_list =', shift_list)
		print('train_op_list =', train_op_list)
		print('test_op_list =', test_op_list)

		overlap = int(sequence_with_overlap)

		# Training set
		train_data = {'pianoroll': np.concatenate([corpus_aug_reshape[shift_id][op]['pianoroll'][overlap] for shift_id in shift_list for op in train_op_list], axis=0),
									'tonal_centroid': np.concatenate([corpus_aug_reshape[shift_id][op]['tonal_centroid'][overlap] for shift_id in shift_list for op in train_op_list], axis=0),
									'len': np.concatenate([corpus_aug_reshape[shift_id][op]['len'][overlap] for shift_id in shift_list for op in train_op_list], axis=0),
									'label': np.concatenate([corpus_aug_reshape[shift_id][op]['label'][overlap] for shift_id in shift_list for op in train_op_list], axis=0)}

		train_data_label_key = np.zeros_like(train_data['label'], dtype=np.int32)
		train_data_label_degree1 = np.zeros_like(train_data['label'], dtype=np.int32)
		train_data_label_degree2 = np.zeros_like(train_data['label'], dtype=np.int32)
		train_data_label_quality = np.zeros_like(train_data['label'], dtype=np.int32)
		train_data_label_inversion = train_data['label']['inversion']

		# Functional harmony labels
		'''key: 42'''
		for k, v in key_dict.items():
				train_data_label_key[train_data['label']['key'] == k] = v
		'''degree1: 9'''
		for k, v in degree1_dict.items():
				train_data_label_degree1[train_data['label']['degree1'] == k] = v
		'''degree2: 14'''
		for k, v in degree2_dict.items():
				train_data_label_degree2[train_data['label']['degree2'] == k] = v
		'''quality: 10'''
		for k, v in quality_dict.items():
				train_data_label_quality[train_data['label']['quality'] == k] = v
		'''inversion: 4'''
		train_data_label_inversion[train_data_label_inversion == -1] = 4
		'''roman numeral: (degree1, degree2, quality, inversion)'''
		train_data_label_roman = train_data_label_degree1 * 14 * 10 * 4 + train_data_label_degree2 * 10 * 4 + train_data_label_quality * 4 + train_data_label_inversion
		train_data_label_roman[train_data['label']['key'] == 'pad'] = 9 * 14 * 10 * 4

		train_data['key'] = train_data_label_key
		train_data['degree1'] = train_data_label_degree1
		train_data['degree2'] = train_data_label_degree2
		train_data['quality'] = train_data_label_quality
		train_data['inversion'] = train_data_label_inversion
		train_data['roman'] = train_data_label_roman

		# Test set
		test_data = {'pianoroll': np.concatenate([corpus_aug_reshape['shift_0'][op]['pianoroll'][0] for op in test_op_list], axis=0),
								 'tonal_centroid': np.concatenate([corpus_aug_reshape['shift_0'][op]['tonal_centroid'][0] for op in test_op_list], axis=0),
								 'len': np.concatenate([corpus_aug_reshape['shift_0'][op]['len'][0] for op in test_op_list], axis=0),
								 'label': np.concatenate([corpus_aug_reshape['shift_0'][op]['label'][0] for op in test_op_list], axis=0)}

		test_data_label_key = np.zeros_like(test_data['label'], dtype=np.int32)
		test_data_label_degree1 = np.zeros_like(test_data['label'], dtype=np.int32)
		test_data_label_degree2 = np.zeros_like(test_data['label'], dtype=np.int32)
		test_data_label_quality = np.zeros_like(test_data['label'], dtype=np.int32)
		test_data_label_inversion = test_data['label']['inversion']

		# Functional harmony labels
		'''key: 42'''
		for k, v in key_dict.items():
				test_data_label_key[test_data['label']['key'] == k] = v
		'''degree1: 9'''
		for k, v in degree1_dict.items():
				test_data_label_degree1[test_data['label']['degree1'] == k] = v
		'''degree2: 14'''
		for k, v in degree2_dict.items():
				test_data_label_degree2[test_data['label']['degree2'] == k] = v
		'''quality: 10'''
		for k, v in quality_dict.items():
				test_data_label_quality[test_data['label']['quality'] == k] = v
		'''inversion: 4'''
		test_data_label_inversion[test_data_label_inversion == -1] = 4
		'''roman numeral'''
		test_data_label_roman = test_data_label_degree1 * 14 * 10 * 4 + test_data_label_degree2 * 10 * 4 + test_data_label_quality * 4 + test_data_label_inversion
		test_data_label_roman[test_data['label']['key'] == 'pad'] = 9 * 14 * 10 * 4

		test_data['key'] = test_data_label_key
		test_data['degree1'] = test_data_label_degree1
		test_data['degree2'] = test_data_label_degree2
		test_data['quality'] = test_data_label_quality
		test_data['inversion'] = test_data_label_inversion
		test_data['roman'] = test_data_label_roman

		print('keys in train/test_data =', train_data.keys())
		return train_data, test_data

def compute_pre_PRF(predicted, actual):
		predicted = tf.cast(predicted, tf.float32)
		actual = tf.cast(actual, tf.float32)
		TP = tf.count_nonzero(predicted * actual, dtype=tf.float32)
		# TN = tf.count_nonzero((predicted - 1) * (actual - 1), dtype=tf.float32)
		FP = tf.count_nonzero(predicted * (actual - 1), dtype=tf.float32)
		FN = tf.count_nonzero((predicted - 1) * actual, dtype=tf.float32)
		return TP, FP, FN

def comput_PRF_with_pre(TP, FP, FN):
		precision = TP / (TP + FP)
		recall = TP / (TP + FN)
		F1 = 2 * precision * recall / (precision + recall)
		precision = tf.cond(tf.is_nan(precision), lambda: tf.constant(0.0), lambda: precision)
		recall = tf.cond(tf.is_nan(recall), lambda: tf.constant(0.0), lambda: recall)
		F1 = tf.cond(tf.is_nan(F1), lambda: tf.constant(0.0), lambda: F1)
		return precision, recall, F1

def save_data(save_dir, data, pred_cc_values, pred_k_values, pred_r_values, eval_mask_values):
		os.makedirs(save_dir, exist_ok=True)

		data_file = os.path.join(save_dir, 'data.pickle')
		with open(data_file, 'wb') as f:
				pickle.dump(data, f)
				print("Saved", data_file)
		
		pred_cc_file = os.path.join(save_dir, 'pred_cc_values.pickle')
		with open(pred_cc_file, 'wb') as f:
				pickle.dump(pred_cc_values, f)
				print("Saved", pred_cc_file)
		
		pred_k_file = os.path.join(save_dir, 'pred_k_values.pickle')
		with open(pred_k_file, 'wb') as f:
				pickle.dump(pred_k_values, f)
				print("Saved", pred_k_file)
				
		pred_r_file = os.path.join(save_dir, 'pred_r_values.pickle')
		with open(pred_r_file, 'wb') as f:
				pickle.dump(pred_r_values, f)
				print("Saved", pred_r_file)
				
		eval_mask_file = os.path.join(save_dir, 'eval_mask_values.pickle')
		with open(eval_mask_file, 'wb') as f:
				pickle.dump(eval_mask_values, f)
				print("Saved", eval_mask_file)

def train_HT():
		print('Run HT functional harmony recognition on %s-%d...' % (hp.dataset, hp.test_set_id))

		# Load training and testing data
		train_data, test_data = load_data_functional(dir=hp.dataset + '_preprocessed_data_MIREX_Mm.pickle', test_set_id=hp.test_set_id, sequence_with_overlap=hp.train_sequence_with_overlap)
		n_train_sequences = train_data['pianoroll'].shape[0]
		n_test_sequences = test_data['pianoroll'].shape[0]
		n_iterations_per_epoch = int(math.ceil(n_train_sequences/hp.n_batches))
		print('n_train_sequences =', n_train_sequences)
		print('n_test_sequences =', n_test_sequences)
		print('n_iterations_per_epoch =', n_iterations_per_epoch)
		print(hp)

		with tf.name_scope('placeholder'):
				x_p = tf.placeholder(tf.int32, [None, hp.n_steps, 88], name="pianoroll")
				x_len = tf.placeholder(tf.int32, [None], name="seq_lens")
				y_k = tf.placeholder(tf.int32, [None, hp.n_steps], name="key") # 7 degrees * 3 accidentals * 2 modes = 42
				y_r = tf.placeholder(tf.int32, [None, hp.n_steps], name="roman_numeral")
				y_cc = tf.placeholder(tf.int32, [None, hp.n_steps], name="chord_change")
				dropout = tf.placeholder(dtype=tf.float32, name="dropout_rate")
				is_training = tf.placeholder(dtype=tf.bool, name="is_training")
				global_step = tf.placeholder(dtype=tf.int32, name='global_step')
				slope = tf.placeholder(dtype=tf.float32, name='annealing_slope')

		with tf.name_scope('model'):
				x_in = tf.cast(x_p, tf.float32)
				source_mask = tf.sequence_mask(lengths=x_len, maxlen=hp.n_steps, dtype=tf.float32) # [n_batches, n_steps]
				target_mask = source_mask
				# chord_change_logits, dec_input_embed, enc_weights, dec_weights = crm.HT(x_in, source_mask, target_mask, slope, dropout, is_training, hp)
				chord_change_logits, dec_input_embed, enc_weights, dec_weights, _, _ = crm.HTv2(x_in, source_mask, target_mask, slope, dropout, is_training, hp)

		with tf.variable_scope("output_projection"):
				n_key_classes = 42 + 1
				n_roman_classes = 9 * 14 * 10 * 4 + 1
				dec_input_embed = tf.layers.dropout(dec_input_embed, rate=dropout, training=is_training)
				key_logits = tf.layers.dense(dec_input_embed, n_key_classes)
				roman_logits = tf.layers.dense(dec_input_embed, n_roman_classes)

		with tf.name_scope('loss'):
				# Chord change
				loss_cc = 4 * tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.cast(y_cc, tf.float32), logits=slope*chord_change_logits, weights=source_mask)
				# Key
				loss_k = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(y_k, n_key_classes), logits=key_logits, weights=target_mask, label_smoothing=0.01)
				# Roman numeral
				loss_r = 0.5 * tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(y_r, n_roman_classes), logits=roman_logits, weights=target_mask, label_smoothing=0.0)
				# Total loss
				loss = loss_cc + loss_k + loss_r
		valid = tf.reduce_sum(target_mask)
		summary_loss = tf.Variable([0.0 for _ in range(4)], trainable=False, dtype=tf.float32)
		summary_valid = tf.Variable(0.0, trainable=False, dtype=tf.float32)
		update_loss = tf.assign(summary_loss, summary_loss + valid * [loss, loss_cc, loss_k, loss_r])
		update_valid = tf.assign(summary_valid, summary_valid + valid)
		mean_loss = tf.assign(summary_loss, summary_loss / summary_valid)
		clr_summary_loss = summary_loss.initializer
		clr_summary_valid = summary_valid.initializer
		tf.summary.scalar('Loss_total', summary_loss[0])
		tf.summary.scalar('Loss_chord_change', summary_loss[1])
		tf.summary.scalar('Loss_key', summary_loss[2])
		tf.summary.scalar('Loss_roman', summary_loss[3])

		with tf.name_scope('evaluation'):
				eval_mask = tf.cast(target_mask, tf.bool)
				# Chord change
				pred_cc = tf.cast(tf.round(tf.sigmoid(slope*chord_change_logits)), tf.int32)
				pred_cc_mask = tf.boolean_mask(pred_cc, tf.cast(source_mask, tf.bool))
				y_cc_mask = tf.boolean_mask(y_cc, tf.cast(source_mask, tf.bool))
				TP_cc, FP_cc, FN_cc = compute_pre_PRF(pred_cc_mask, y_cc_mask)
				# Key
				pred_k = tf.argmax(key_logits, axis=2, output_type=tf.int32)
				pred_k_correct = tf.equal(pred_k, y_k)
				pred_k_correct_mask = tf.boolean_mask(tensor=pred_k_correct, mask=eval_mask)
				n_correct_k = tf.reduce_sum(tf.cast(pred_k_correct_mask, tf.float32))
				# Roman numeral
				pred_r = tf.argmax(roman_logits, axis=2, output_type=tf.int32)
				pred_r_correct = tf.equal(pred_r, y_r)
				pred_r_correct_mask = tf.boolean_mask(tensor=pred_r_correct, mask=eval_mask)
				n_correct_r = tf.reduce_sum(tf.cast(pred_r_correct_mask, tf.float32))
				n_total = tf.cast(tf.size(pred_r_correct_mask), tf.float32)
		summary_count = tf.Variable([0.0 for _ in range(6)], trainable=False, dtype=tf.float32)
		summary_score = tf.Variable([0.0 for _ in range(5)], trainable=False, dtype=tf.float32)
		update_count = tf.assign(summary_count, summary_count + [n_correct_k, n_correct_r, n_total, TP_cc, FP_cc, FN_cc])
		acc_k = summary_count[0] / summary_count[2]
		acc_r = summary_count[1] / summary_count[2]
		P_cc, R_cc, F1_cc = comput_PRF_with_pre(summary_count[3], summary_count[4], summary_count[5])
		update_score = tf.assign(summary_score, summary_score + [acc_k, acc_r, P_cc, R_cc, F1_cc])
		clr_summary_count = summary_count.initializer
		clr_summary_score = summary_score.initializer
		tf.summary.scalar('Accuracy_key', summary_score[0])
		tf.summary.scalar('Accuracy_roman', summary_score[1])
		tf.summary.scalar('Precision_cc', summary_score[2])
		tf.summary.scalar('Recall_cc', summary_score[3])
		tf.summary.scalar('F1_cc', summary_score[4])

		with tf.name_scope('optimization'):
				# Apply warn-up learning rate
				warm_up_steps = tf.constant(4000, dtype=tf.float32)
				gstep = tf.cast(global_step, dtype=tf.float32)
				learning_rate = pow(hp.input_embed_size, -0.5) * tf.minimum(tf.pow(gstep, -0.5), gstep * tf.pow(warm_up_steps, -1.5))
				optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
																					 beta1=0.9,
																					 beta2=0.98,
																					 epsilon=1e-9)
				train_op = optimizer.minimize(loss)
		# Graph location and summary writers
		# print('Saving graph to: %s' % hp.graph_location)
		# merged = tf.summary.merge_all()
		# train_writer = tf.summary.FileWriter(hp.graph_location + '/train')
		# test_writer = tf.summary.FileWriter(hp.graph_location + '/test')
		# train_writer.add_graph(tf.get_default_graph())
		# test_writer.add_graph(tf.get_default_graph())
		saver = tf.train.Saver(max_to_keep=1)

		def predict():
			with tf.Session() as sess:
				saver.restore(sess, 'model_results/HT_functional_harmony_recognition_' + hp.dataset + '_' + str(hp.test_set_id) + '.ckpt')
				print("Model restored.")

				process_list = ['v']#[x for x in list(string.ascii_lowercase) if x not in ['e', 'i', 'k', 'n', 'q', 'u', 'x', 'y', 'z']] 
				for c in process_list:
					filepath = f"/home/ubuntu/project/Harmony-Transformer-v2/preprocessed/datasets_{c}_MIREX_Mm/corpus_aug_reshape.pickle"
					print(f"PROCESSING: {filepath}")

					data = load_data_unlabeled(filepath)
					print("Data loaded from", filepath)

					# for chunk_idx, data in enumerate(data_chunks):
					feed_dict = {
							x_p: data['pianoroll'],
							x_len: data['len'],
							dropout: 0.0,
							is_training: False,
							slope: 1.771561000000001 # best annealing slope from the saved train run
					}

					# Define the operations to run (predictions)
					predictions = [pred_cc, pred_k, pred_r, eval_mask]

					# Run the session to get predictions
					pred_cc_values, pred_k_values, pred_r_values, eval_mask_values = sess.run(predictions, feed_dict=feed_dict)

					print('Chord Change Predictions:', pred_cc_values)
					print('Key Predictions:', pred_k_values)
					print('Roman Numeral Predictions:', pred_r_values)
					print('Evaluation Mask:', eval_mask_values)

					parent_dir = os.path.dirname(filepath)
					# chunk_dir = os.path.join(parent_dir, f"chunk_{chunk_idx}")
					save_data(parent_dir, data, pred_cc_values, pred_k_values, pred_r_values, eval_mask_values)
					print(f"FINISHED PROCESSING: {filepath}")
			return 

		predict()
		sys.exit(0)
		# ------------------------------ STOP HERE FOR UNLABELED DATA PREDICTION --------------------------------
		
		# Training
		print('Train the model...')
		with tf.Session() as sess:
				sess.run(tf.global_variables_initializer())
				startTime = time.time() # start time of training
				best_score = [0.0 for _ in range(6)]
				in_succession = 0
				best_epoch = 0
				annealing_slope = 1.0
				best_slope = 0.0
				for step in range(hp.n_training_steps):
						# Training
						if step == 0:
								indices = range(n_train_sequences)
								batch_indices = [indices[x:x + hp.n_batches] for x in range(0, len(indices), hp.n_batches)]

						if step > 0 and step % n_iterations_per_epoch == 0:
								annealing_slope *= hp.annealing_rate

						if step >= 2*n_iterations_per_epoch and step % n_iterations_per_epoch == 0:
								# Shuffle training data
								indices = random.sample(range(n_train_sequences), n_train_sequences)
								batch_indices = [indices[x:x + hp.n_batches] for x in range(0, len(indices), hp.n_batches)]

						batch = (train_data['pianoroll'][batch_indices[step % len(batch_indices)]],
										 train_data['len'][batch_indices[step % len(batch_indices)]],
										 train_data['label']['chord_change'][batch_indices[step % len(batch_indices)]],
										 train_data['key'][batch_indices[step % len(batch_indices)]],
										 train_data['roman'][batch_indices[step % len(batch_indices)]],
										 train_data['degree1'][batch_indices[step % len(batch_indices)]],
										 train_data['degree2'][batch_indices[step % len(batch_indices)]],
										 train_data['quality'][batch_indices[step % len(batch_indices)]],
										 train_data['inversion'][batch_indices[step % len(batch_indices)]],
										 train_data['label']['key'][batch_indices[step % len(batch_indices)]])

						train_run_list = [train_op, update_valid, update_loss, update_count, loss, loss_cc, loss_k, loss_r, pred_cc, pred_k, pred_r, eval_mask, enc_weights, dec_weights]
						train_feed_fict = {x_p: batch[0],
															 x_len: batch[1],
															 y_cc: batch[2],
															 y_k: batch[3],
															 y_r: batch[4],
															 dropout: hp.drop,
															 is_training: True,
															 global_step: step + 1,
															 slope: annealing_slope}
						_, _, _, _, train_loss, train_loss_cc, train_loss_k, train_loss_r, \
						train_pred_cc, train_pred_k, train_pred_r, train_eval_mask, enc_w, dec_w = sess.run(train_run_list, feed_dict=train_feed_fict)
						if step == 0:
								print('*~ loss_cc %.4f, loss_k %.4f, loss_r %.4f ~*' % (train_loss_cc, train_loss_k, train_loss_r))

						# Display training log & Testing
						if step > 0 and step % n_iterations_per_epoch == 0:
								sess.run([mean_loss, update_score])
								train_summary, train_loss, train_score = sess.run([merged, summary_loss, summary_score])
								sess.run([clr_summary_valid, clr_summary_loss, clr_summary_count, clr_summary_score])
								train_writer.add_summary(train_summary, step)
								print("---- step %d, epoch %d: train_loss: total %.4f (cc %.4f, k %.4f, r %.4f), evaluation: k %.4f, r %.4f, cc (P %.4f, R %.4f, F1 %.4f) ----"
										% (step, step // n_iterations_per_epoch, train_loss[0], train_loss[1], train_loss[2], train_loss[3],
											 train_score[0], train_score[1], train_score[2], train_score[3], train_score[4]))
								print('enc_w =', enc_w, 'dec_w =', dec_w)
								display_len = 32
								n_just = 5
								print('len =', batch[1][0])
								print('y_k'.ljust(7, ' '), ''.join([b.rjust(n_just, ' ') for b in batch[9][0, :display_len]]))
								print('y_d1'.ljust(7, ' '), ''.join([[k for k, v in degree1_dict.items() if v == b][0].rjust(n_just, ' ') for b in batch[5][0, :display_len]]))
								print('y_d2'.ljust(7, ' '), ''.join([[k for k, v in degree2_dict.items() if v == b][0].rjust(n_just, ' ') for b in batch[6][0, :display_len]]))
								print('y_q'.ljust(7, ' '), ''.join([[k for k, v in quality_dict.items() if v == b][0].rjust(n_just, ' ') for b in batch[7][0, :display_len]]))
								print('y_inv'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in batch[8][0, :display_len]]))
								print('valid'.ljust(7, ' '), ''.join(['y'.rjust(n_just, ' ') if b else 'n'.rjust(n_just, ' ') for b in train_eval_mask[0, :display_len]]))
								print('y_cc'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in batch[2][0, :display_len]]))
								print('pred_cc'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in train_pred_cc[0, :display_len]]))
								print('y_k'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in batch[3][0, :display_len]]))
								print('pred_k'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in train_pred_k[0, :display_len]]))
								print('y_r'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in batch[4][0, :display_len]]))
								print('pred_r'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in train_pred_r[0, :display_len]]))

								# Testing
								test_run_list = [update_valid, update_loss, update_count, pred_cc, pred_k, pred_r, eval_mask]
								test_feed_fict = {x_p: test_data['pianoroll'],
																	x_len: test_data['len'],
																	y_cc: test_data['label']['chord_change'],
																	y_k: test_data['key'],
																	y_r: test_data['roman'],
																	dropout: 0.0,
																	is_training: False,
																	slope: annealing_slope}
								_, _, _, test_pred_cc, test_pred_k, test_pred_r, test_eval_mask = sess.run(test_run_list, feed_dict=test_feed_fict)
								sess.run([mean_loss, update_score])
								test_summary, test_loss, test_score = sess.run([merged, summary_loss, summary_score])
								sess.run([clr_summary_valid, clr_summary_loss, clr_summary_count, clr_summary_score])
								test_writer.add_summary(test_summary, step)

								sq = crm.segmentation_quality(test_data['roman'], test_pred_r, test_data['len'])
								print("==== step %d, epoch %d: test_loss: total %.4f (cc %.4f, k %.4f, r %.4f), evaluation: k %.4f, r %.4f, cc (P %.4f, R %.4f, F1 %.4f), sq %.4f ===="
											% (step, step // n_iterations_per_epoch, test_loss[0], test_loss[1], test_loss[2], test_loss[3],
												 test_score[0], test_score[1], test_score[2], test_score[3], test_score[4], sq))
								sample_id = random.randint(0, n_test_sequences - 1)
								print('len =', test_data['len'][sample_id])
								print('y_k'.ljust(7, ' '), ''.join([b.rjust(n_just, ' ') for b in test_data['label']['key'][sample_id, :display_len]]))
								print('y_d1'.ljust(7, ' '), ''.join([[k for k, v in degree1_dict.items() if v == b][0].rjust(n_just, ' ') for b in test_data['degree1'][sample_id, :display_len]]))
								print('y_d2'.ljust(7, ' '), ''.join([[k for k, v in degree2_dict.items() if v == b][0].rjust(n_just, ' ') for b in test_data['degree2'][sample_id, :display_len]]))
								print('y_q'.ljust(7, ' '), ''.join([[k for k, v in quality_dict.items() if v == b][0].rjust(n_just, ' ') for b in test_data['quality'][sample_id, :display_len]]))
								print('y_inv'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in test_data['inversion'][sample_id, :display_len]]))
								print('valid'.ljust(7, ' '), ''.join(['y'.rjust(n_just, ' ') if b else 'n'.rjust(n_just, ' ') for b in test_eval_mask[sample_id, :display_len]]))
								print('y_cc'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in test_data['label']['chord_change'][sample_id, :display_len]]))
								print('pred_cc'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in test_pred_cc[sample_id, :display_len]]))
								print('y_k'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in test_data['key'][sample_id, :display_len]]))
								print('pred_k'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in test_pred_k[sample_id, :display_len]]))
								print('y_r'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in test_data['roman'][sample_id, :display_len]]))
								print('pred_r'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in test_pred_r[sample_id, :display_len]]))

								if step > 0 and sum(test_score[:2]) > sum(best_score[:2]):
										best_score = np.concatenate([test_score, [sq]], axis=0)
										best_epoch = step // n_iterations_per_epoch
										best_slope = annealing_slope
										in_succession = 0
										# Save variables of the model
										print('*saving variables...\n')
										saver.save(sess, hp.graph_location + '/HT_functional_harmony_recognition_' + hp.dataset + '_' + str(hp.test_set_id) + '.ckpt')
								else:
										in_succession += 1
										if in_succession > hp.n_in_succession:
												print('Early stopping.')
												break

				elapsed_time = time.time() - startTime
				print('\nHT functional harmony recognition on %s-%d:' % (hp.dataset, hp.test_set_id))
				print('training time = %.2f hr' % (elapsed_time / 3600))
				print('best epoch = ', best_epoch)
				print('best score =', np.round(best_score, 4))
				print('best slope =', best_slope)

def main():
		train_HT()
		dir_path = '/home/ubuntu/project/Harmony-Transformer-v2/preprocessed/datasets_v_MIREX_Mm'

		with open(os.path.join(dir_path, "pred_cc_values.pickle"), 'rb') as file:
			pred_cc_values = pickle.load(file)
		
		with open(os.path.join(dir_path, "pred_k_values.pickle"), 'rb') as file:
			pred_k_values = pickle.load(file)

		with open(os.path.join(dir_path, "pred_r_values.pickle"), 'rb') as file:
			pred_r_values = pickle.load(file)

		with open(os.path.join(dir_path, "eval_mask_values.pickle"), 'rb') as file:
			eval_mask_values = pickle.load(file)

		with open(os.path.join(dir_path, "data.pickle"), 'rb') as file:
			data = pickle.load(file)
		
		predictions = translate_predictions(data, pred_cc_values, pred_k_values, pred_r_values, eval_mask_values)

		# Print the translated predictions
		for prediction in predictions:
				print(prediction)

if __name__ == '__main__':
		# Hyperparameters
		hyperparameters = namedtuple('hyperparameters',
																 ['dataset',
																	'test_set_id',
																	'graph_location',
																	'n_steps',
																	'input_embed_size',
																	'n_layers',
																	'n_heads',
																	'train_sequence_with_overlap',
																	'initial_learning_rate',
																	'drop',
																	'n_batches',
																	'n_training_steps',
																	'n_in_succession',
																	'annealing_rate'])

		hp = hyperparameters(dataset='BPS_FH', # {'BPS_FH', 'Preludes'}
												 test_set_id=1,
												 graph_location='model',
												 n_steps=128,
												 input_embed_size=128,
												 n_layers=2,
												 n_heads=4,
												 train_sequence_with_overlap=True,
												 initial_learning_rate=1e-4,
												 drop=0.1,
												 n_batches=40,
												 n_training_steps=100000,
												 n_in_succession=10,
												 annealing_rate=1.1)

		main()

