DATA_AUG_SWITCH = True
BINARY_SWITCH = False
KFOLD_SWITCH = True
class_num = 5
file_no = 3
label_range = (2, 3, 4, 5, 6)
loss_min = 0.0001

def set_mode(experiment_mode):
    global DATA_AUG_SWITCH, BINARY_SWITCH, KFOLD_SWITCH, class_num, file_no, label_range
    if experiment_mode == 'binary_mode':
        DATA_AUG_SWITCH = False
        BINARY_SWITCH = True
        KFOLD_SWITCH = True
        class_num = 2
        file_no = 12
        label_range = (1, 2, 3, 4, 5, 6)
    elif experiment_mode == 'five_grade_mode':
        DATA_AUG_SWITCH = True
        BINARY_SWITCH = False
        KFOLD_SWITCH = True
        class_num = 5
        file_no = 3
        label_range = (2, 3, 4, 5, 6)
        loss_min = 0.000005
    elif experiment_mode == 'five_reg_mode':
        DATA_AUG_SWITCH = True
        BINARY_SWITCH = False
        KFOLD_SWITCH = True
        class_num = 1
        file_no = 3
        loss_min = 0.000005
        label_range = (2, 3, 4, 5, 6)

