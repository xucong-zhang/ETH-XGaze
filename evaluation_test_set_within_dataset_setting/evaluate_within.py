#!/usr/bin/env python
import sys, os, os.path
import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt

radians_to_degrees = 180.0 / np.pi

def pitchyaw_to_vector(pitchyaws):
	r"""Convert given yaw (:math:`\theta`) and pitch (:math:`\phi`) angles to unit gaze vectors.

	Args:
		pitchyaws (:obj:`numpy.array`): yaw and pitch angles :math:`(n\times 2)` in radians.

	Returns:
		:obj:`numpy.array` of shape :math:`(n\times 3)` with 3D vectors per row.
	"""
	n = pitchyaws.shape[0]
	sin = np.sin(pitchyaws)
	cos = np.cos(pitchyaws)
	out = np.empty((n, 3))
	out[:, 0] = np.multiply(cos[:, 0], sin[:, 1])
	out[:, 1] = sin[:, 0]
	out[:, 2] = np.multiply(cos[:, 0], cos[:, 1])
	return out

def angular_error(a, b):
	"""Calculate angular error (via cosine similarity)."""
	a = pitchyaw_to_vector(a) if a.shape[1] == 2 else a
	b = pitchyaw_to_vector(b) if b.shape[1] == 2 else b

	ab = np.sum(np.multiply(a, b), axis=1)
	a_norm = np.linalg.norm(a, axis=1)
	b_norm = np.linalg.norm(b, axis=1)

	# Avoid zero-values (to avoid NaNs)
	a_norm = np.clip(a_norm, a_min=1e-7, a_max=None)
	b_norm = np.clip(b_norm, a_min=1e-7, a_max=None)

	similarity = np.divide(ab, np.multiply(a_norm, b_norm))

	return np.arccos(similarity) * radians_to_degrees

if __name__== "__main__":
	input_dir = sys.argv[1]
	output_dir = os.path.join(input_dir, 'output')
	submit_dir = os.path.join(input_dir, 'res')
	truth_dir = os.path.join(input_dir, 'ref')
	
	print('now we begin')
	if not os.path.isdir(submit_dir):
		print("%s doesn't exist" % submit_dir)

	if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)

	output_filename = os.path.join(output_dir, 'scores.txt')
	output_file = open(output_filename, 'w')

	print('loading truth_file')
	truth_file = os.path.join(truth_dir, "reference_within/within_test_labels.txt")
	truth = np.loadtxt(truth_file, delimiter=',')
	index_file = os.path.join(truth_dir, "reference_within/index.txt")
	num_index = np.loadtxt(index_file, dtype=np.int)

	# gold_list = os.listdir(submit_dir)
	# for gold in gold_list:
	# 	print('submit_dir: ', gold)

	print('loading submission file')
	submission_answer_file = os.path.join(submit_dir, "within_eva_results.txt")
	submission_answer = np.loadtxt(submission_answer_file, delimiter=',')
	submission_answer = submission_answer[num_index, :]

	print('now compute the gaze error')
	error_all = angular_error(submission_answer, truth)
	error = np.mean(error_all)
	output_file.write("gaze_error: %0.4f\n" % error)
	print('gaze_error: ', error)
	error_std = np.std(error_all)
	print('gaze_error_std: ', error_std)
	output_file.write("gaze_error_std: %0.4f\n" % error_std)

	output_file.close()

	# output the error distribution
	data_index_file = os.path.join(truth_dir, "reference_within/within_test_head_pose_index.pkl")
	file = open(data_index_file, 'rb')
	show_head_horizon_index = pickle.load(file)
	show_head_ver_index = pickle.load(file)
	show_gaze_horizon_index = pickle.load(file)
	show_gaze_ver_index = pickle.load(file)

	interval = 50
	show_head_horizon_error = np.zeros((interval))
	show_head_ver_error = np.zeros((interval))
	show_gaze_horizon_error = np.zeros((interval))
	show_gaze_ver_error = np.zeros((interval))

	for num_x in range(0, interval):
		if len(show_head_horizon_index[num_x])>1:
			show_head_horizon_error[num_x] = np.median(error_all[show_head_horizon_index[num_x]])
		else:
			show_head_horizon_error[num_x] = np.nan

		if len(show_head_ver_index[num_x]) > 1:
			show_head_ver_error[num_x] = np.median(error_all[show_head_ver_index[num_x]])
		else:
			show_head_ver_error[num_x] = np.nan

		if len(show_gaze_horizon_index[num_x])>1:
			show_gaze_horizon_error[num_x] = np.median(error_all[show_gaze_horizon_index[num_x]])
		else:
			show_gaze_horizon_error[num_x] = np.nan

		if len(show_head_horizon_index[num_x])>1:
			show_gaze_ver_error[num_x] = np.median(error_all[show_gaze_ver_index[num_x]])
		else:
			show_gaze_ver_error[num_x] = np.nan

	save_name = os.path.join(output_dir, 'show_head_horizontal_error.txt')
	print('save the gaze estimation error across horizontal head poses in the file: ', save_name)
	np.savetxt(save_name, show_head_horizon_error)

	save_name = os.path.join(output_dir, 'show_head_vertical_error.txt')
	print('save the gaze estimation error across vertical head poses in the file: ', save_name)
	np.savetxt(save_name, show_head_ver_error)

	save_name = os.path.join(output_dir, 'show_gaze_horizontal_error.txt')
	print('save the gaze estimation error across horizontal gaze directions in the file: ', save_name)
	np.savetxt(save_name, show_gaze_horizon_error)

	save_name = os.path.join(output_dir, 'show_gaze_vertical_error.txt')
	print('save the gaze estimation error across vertical gaze directions in the file: ', save_name)
	np.savetxt(save_name, show_gaze_ver_error)

	# show_x = range(0, interval)
	# show_x = np.asarray(show_x)
	# plt.close('all')
	# # plt.figure(figsize=(12.0, 8.0))
	#
	# # show_y1 = zero_to_nan(show_y1)
	# show_head_horizon_error = np.asarray(show_head_horizon_error)
	# pick_index = np.argwhere(~np.isnan(show_head_horizon_error)).reshape(-1)
	# plt.plot(show_x[pick_index], show_head_horizon_error[pick_index], 'y-', markersize=10, label='Head pose horizontal')
	#
	#
	# # show_y2 = zero_to_nan(show_y2)
	# show_head_ver_error = np.asarray(show_head_ver_error)
	# pick_index = np.argwhere(~np.isnan(show_head_ver_error)).reshape(-1)
	# plt.plot(show_x[pick_index], show_head_ver_error[pick_index], 'm-', markersize=10, label='Head pose vertical')
	#
	#
	# # show_y3 = zero_to_nan(show_y3)
	# show_gaze_horizon_error = np.asarray(show_gaze_horizon_error)
	# pick_index = np.argwhere(~np.isnan(show_gaze_horizon_error)).reshape(-1)
	# plt.plot(show_x[pick_index], show_gaze_horizon_error[pick_index], 'g-', markersize=10, label='Gaze horizontal')
	#
	# # show_y4 = zero_to_nan(show_y4)
	# show_gaze_ver_error = np.asarray(show_gaze_ver_error)
	# pick_index = np.argwhere(~np.isnan(show_gaze_ver_error)).reshape(-1)
	# plt.plot(show_x[pick_index], show_gaze_ver_error[pick_index], 'c-', markersize=10, label='Gaze vertical')
	#
	# plt.legend(fontsize=20)
	#
	# plt.xlim([0, interval])
	# plt.xticks([0, 12.5, 25, 37.5, 50], ['-80', '-40', '0', '40', '80'])
	#
	# # ax.set_ylim([0, 100])
	# plt.ylabel('Gaze Error [degree]')
	#
	# save_name = os.path.join(output_dir, 'error_dis.png')
	# plt.savefig(save_name, bbox_inches='tight', pad_inches=0)
	# print('save image: ', save_name)


