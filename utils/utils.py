import numpy as np
import os
from glob import glob
from utils.csv_process import process_image
from utils.scanpath_utils import remove_duplicates

def get_gt_files(imgname: str, gt_path: str) -> list:
    img_name, task_type = process_image()
    # print(img_name, task_type)
    if imgname not in img_name['filename'].to_list(): return None
    img_id = img_name[img_name['filename'] == imgname]['imageID'].to_numpy()[0]
    gt_files_noclass = glob(os.path.join(gt_path, f'*fix_{img_id}.tsv'))
    gt_a = []
    gt_b = []
    gt_c = []
    for gt_file in gt_files_noclass:
        p_name = gt_file.split('/')[-1].split('_')[-3].replace('p','P')
        taskT = task_type.at[p_name,str(img_id)]
        if taskT == 'A':
            gt_a.append(gt_file)
        elif taskT == 'B':
            gt_b.append(gt_file)
        elif taskT == 'C':
            gt_c.append(gt_file)
        else:
            print('Error: Task Type not found')
    return [gt_a, gt_b, gt_c]

def get_gt_strings(gtpath:str, imname:str, is_simplified: False) -> list:
	STR_a=[]
	STR_b=[]
	STR_c=[]
	gtstrings_a = glob(os.path.join(gtpath,imname,'A','*.txt'))
	gtstrings_b = glob(os.path.join(gtpath,imname,'B','*.txt'))
	gtstrings_c = glob(os.path.join(gtpath,imname,'C','*.txt'))
	for gtstr in gtstrings_a:
		with open(gtstr,'r',encoding='utf-8') as f:
			line = f.readline()
		if is_simplified:
			line = remove_duplicates(line)
		STR_a.append(line)
	for gtstr in gtstrings_b:
		with open(gtstr,'r',encoding='utf-8') as f:
			line = f.readline()
		if is_simplified:
			line = remove_duplicates(line)
		STR_b.append(line)
	for gtstr in gtstrings_c:
		with open(gtstr,'r',encoding='utf-8') as f:
			line = f.readline()
		if is_simplified:
			line = remove_duplicates(line)
		STR_c.append(line)
	return [STR_a, STR_b, STR_c]

def scanpath_to_string(scanpath, height, width, Xbins, Ybins, Tbins):
	"""
			a b c d ...
		A
		B
		C
		D

		returns Aa
	"""
	if Tbins !=0:
		try:
			assert scanpath.shape[1] == 3
		except Exception as x:
			print("Temporal information doesn't exist.")

	height_step, width_step = height//Ybins, width//Xbins
	string = ''
	num = list()
	for i in range(scanpath.shape[0]):
		fixation = scanpath[i].astype(np.int32)
		xbin = fixation[0]//width_step
		ybin = ((height - fixation[1])//height_step)
		corrs_x = chr(65 + xbin)
		corrs_y = chr(97 + ybin)
		T = 1
		if Tbins:
			T = fixation[2]//Tbins
		for t in range(T):
			string += (corrs_y + corrs_x)
			num += [(ybin * Xbins) + xbin]
	return string, num


def global_align(P, Q, SubMatrix=None, gap=0, match=1, mismatch=-1):
	"""
		https://bitbucket.org/brentp/biostuff/src/
	"""
	UP, LEFT, DIAG, NONE = range(4)
	max_p = len(P)
	max_q = len(Q)
	score   = np.zeros((max_p + 1, max_q + 1), dtype='f')
	pointer = np.zeros((max_p + 1, max_q + 1), dtype='i')

	pointer[0, 0] = NONE
	score[0, 0] = 0.0
	pointer[0, 1:] = LEFT
	pointer[1:, 0] = UP

	score[0, 1:] = gap * np.arange(max_q)
	score[1:, 0] = gap * np.arange(max_p).T

	for i in range(1, max_p + 1):
		ci = P[i - 1]
		for j in range(1, max_q + 1):
			cj = Q[j - 1]
			if SubMatrix is None:
				diag_score = score[i - 1, j - 1] + (cj == ci and match or mismatch)
			else:
				diag_score = score[i - 1, j - 1] + SubMatrix[cj][ci]
			up_score   = score[i - 1, j] + gap
			left_score = score[i, j - 1] + gap

			if diag_score >= up_score:
				if diag_score >= left_score:
					score[i, j] = diag_score
					pointer[i, j] = DIAG
				else:
					score[i, j] = left_score
					pointer[i, j] = LEFT
			else:
				if up_score > left_score:
					score[i, j ]  = up_score
					pointer[i, j] = UP
				else:
					score[i, j]   = left_score
					pointer[i, j] = LEFT

	align_j = ""
	align_i = ""
	while True:
		p = pointer[i, j]
		if p == NONE: break
		s = score[i, j]
		if p == DIAG:
			# align_j += Q[j - 1]
			# align_i += P[i - 1]
			i -= 1
			j -= 1
		elif p == LEFT:
			# align_j += Q[j - 1]
			# align_i += "-"
			j -= 1
		elif p == UP:
			# align_j += "-"
			# align_i += P[i - 1]
			i -= 1
		else:
			raise ValueError
	# return align_j[::-1], align_i[::-1]
	return score.max()
