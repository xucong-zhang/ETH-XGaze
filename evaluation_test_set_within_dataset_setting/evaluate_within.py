#!/usr/bin/env python
import os, os.path
import numpy as np

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
	input_dir = os.path.dirname(os.path.realpath(__file__)) # taking the current file folder as the working folder
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

	print('loading submission file')
	submission_answer_file = os.path.join(submit_dir, "within_eva_results.txt")
	submission_answer = np.loadtxt(submission_answer_file, delimiter=',')

	print('now compute the gaze error')
	error_all = angular_error(submission_answer, truth)
	error = np.mean(error_all)
	output_file.write("gaze_error: %0.4f\n" % error)
	print('gaze_error: ', error)
	error_std = np.std(error_all)
	print('gaze_error_std: ', error_std)
	output_file.write("gaze_error_std: %0.4f\n" % error_std)

	output_file.close()