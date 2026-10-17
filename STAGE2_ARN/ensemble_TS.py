import argparse
import pickle

import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', default='ntu/xsub', choices={'ntu/xsub'},
                    help='the work folder for storing results')
# parser.add_argument('--datasets', default='ntu/xview', choices={'ntu/xview'},
#                     help='the work folder for storing results')
parser.add_argument('--alpha', default=1, help='weighted summation')
arg = parser.parse_args()

dataset = arg.datasets
label = open('/media/user/FCCA96CFCA9686141/BIN/MS-G3D/data/ntu/xsub/val_label.pkl', 'rb')
label = np.array(pickle.load(label))

# ----------------------------------VA + RICH + MSG3D ----------------------------------- #
r1 = open(r'../teacher_weight/S1/epoch41_test_score.pkl', 'rb')
r2 = open(r'../teacher_weight/S2/epoch46_test_score.pkl', 'rb')
r3 = open(r'../teacher_weight/S3/epoch48_test_score.pkl', 'rb')
r4 = open(r'../teacher_weight/T2/epoch48_test_score.pkl', 'rb')
r5 = open(r'../teacher_weight/T3/epoch44_test_score.pkl', 'rb')

# ----------------------------------J & E & S & D & V(33211)_92.05%----------------------------------- #
# ----------------------------------J & E & D (433)_91.41%-------------------------------------------- #
# ----------------------------------J & E & S & D (3331)_92.24%--------------------------------------- #
# ----------------------------------J & E & D & V(3322)_91.53%---------------------------------------- #

# r1 = open(r'/home/antony/MS-G3D/work_dir/ntu/xsub/LF/JESDV/J/epoch46_test_score.pkl', 'rb')
# r2 = open(r'/home/antony/MS-G3D/work_dir/ntu/xsub/LF/JESDV/E/epoch49_test_score.pkl', 'rb')
# r3 = open(r'/home/antony/MS-G3D/work_dir/ntu/xsub/LF/JESDV/S/epoch48_test_score.pkl', 'rb')
# r4 = open(r'/home/antony/MS-G3D/work_dir/ntu/xsub/LF/JESDV/D/epoch45_test_score.pkl', 'rb')
# r5 = open(r'/home/antony/MS-G3D/work_dir/ntu/xsub/LF/JESDV/V/epoch38_test_score.pkl', 'rb')

# ----------------------------------VA_J & E & S & D & V(3 3.5 1.5 1 1)_92.17%----------------------------------- #
# ----------------------------------VAJ & E & D (433)_91.54%-------------------------------------------- #
# ----------------------------------VAJ & E & S & D (3321)_92.12%--------------------------------------- #
# ----------------------------------VAJ & E & D & V(3322)_91.58%---------------------------------------- #

# r1 = open(r'/home/antony/MS-G3D/work_dir/ntu/xsub/LF/VAJ_ESDV/VAJ/epoch49_test_score.pkl', 'rb')
# r2 = open(r'/home/antony/MS-G3D/work_dir/ntu/xsub/LF/VAJ_ESDV/VAJ_E/epoch41_test_score.pkl', 'rb')
# r3 = open(r'/home/antony/MS-G3D/work_dir/ntu/xsub/LF/VAJ_ESDV/VAJ_S/epoch50_test_score.pkl', 'rb')
# r4 = open(r'/home/antony/MS-G3D/work_dir/ntu/xsub/LF/VAJ_ESDV/VAJ_D/epoch48_test_score.pkl', 'rb')
# r5 = open(r'/home/antony/MS-G3D/work_dir/ntu/xsub/LF/VAJ_ESDV/VAJ_V/epoch48_test_score.pkl', 'rb')

# ----------------------------------J & E & S & D & V(33211)_96.1%----------------------------------- #
# ----------------------------------J & E & D (433)_95.98-------------------------------------------- #
# ----------------------------------J & E & S & D (3322)_96.1%--------------------------------------- #
# ----------------------------------J & E & D & V(3322)_95.87%---------------------------------------- #

# r1 = open(r'/home/antony/MS-G3D/work_dir/ntu/xview/Rich/J_G3D/epoch49_test_score.pkl', 'rb')
# r2 = open(r'/home/antony/MS-G3D/work_dir/ntu/xview/Rich/E_G3D/epoch50_test_score.pkl', 'rb')
# r3 = open(r'/home/antony/MS-G3D/work_dir/ntu/xview/Rich/S_G3D/epoch46_test_score.pkl', 'rb')
# r4 = open(r'/home/antony/MS-G3D/work_dir/ntu/xview/Rich/D_G3D/epoch49_test_score.pkl', 'rb')
# r5 = open(r'/home/antony/MS-G3D/work_dir/ntu/xview/Rich/V_G3D/epoch41_test_score.pkl', 'rb')

# ----------------------------------VA_J & E & S & D & V(3 3.5 1.5 1 1)_96.0%----------------------------------- #
# ----------------------------------VAJ & E & D (433)_95.67-------------------------------------------- #
# ----------------------------------VAJ & E & S & D (3322)_95.96%--------------------------------------- #
# ----------------------------------VAJ & E & D & V(3322)_95.68%---------------------------------------- #

# r1 = open(r'/home/antony/MS-G3D/work_dir/ntu/xview/Rich/VAJ_G3D/epoch44_test_score.pkl', 'rb')
# r2 = open(r'/home/antony/MS-G3D/work_dir/ntu/xview/Rich/VAJ_E_G3D/epoch50_test_score.pkl', 'rb')
# r3 = open(r'/home/antony/MS-G3D/work_dir/ntu/xview/Rich/VAJ_S_G3D/epoch47_test_score.pkl', 'rb')
# r4 = open(r'/home/antony/MS-G3D/work_dir/ntu/xview/Rich/VAJ_D_G3D/epoch38_test_score.pkl', 'rb')
# r5 = open(r'/home/antony/MS-G3D/work_dir/ntu/xview/Rich/VAJ_V_G3D/epoch50_test_score.pkl', 'rb')


r1 = list(pickle.load(r1).items())
r2 = list(pickle.load(r2).items())
r3 = list(pickle.load(r3).items())
r4 = list(pickle.load(r4).items())
r5 = list(pickle.load(r5).items())

r1.sort(key=lambda k: k[0])
r2.sort(key=lambda k: k[0])
r3.sort(key=lambda k: k[0])
r4.sort(key=lambda k: k[0])
r5.sort(key=lambda k: k[0])
label = label.T[np.lexsort(label[::-1, :])].T

right_num = total_num = right_num_5 = 0
for i in tqdm(range(len(label[0]))):
    _, l = label[:, i]
    _, r11 = r1[i]
    _, r22 = r2[i]
    _, r33 = r3[i]
    _, r44 = r4[i]
    _, r55 = r5[i]

    # r = r11*0.4 + r22*0.3 + r44*0.3
    # r = r11*0.3 + r22*0.3 + r33*0.2 + r44*0.2
    # r = r11*0.3 + r22*0.3 + r44*0.2 + r55*0.2
    # r = r11*0.3 + r22*0.35 + r33*0.15 + r44*0.1 + r55*0.1
    r = r11*0.3 + r22*0.3 + r33*0.3 + r44*0.1
    rank_5 = r.argsort()[-5:]
    right_num_5 += int(int(l) in rank_5)
    r = np.argmax(r)
    right_num += int(r == int(l))
    total_num += 1

acc = right_num / total_num
acc5 = right_num_5 / total_num

print('top1_acc :', acc * 100)
print('top5_acc :', acc5 * 100)
