import numpy as np

part_file = '../list_eval_partition.txt'
attr_file = '../list_attr_celeba.txt'
partition = np.loadtxt(part_file, str, comments=None, delimiter='\n')
attribute = np.loadtxt(attr_file, str, comments=None, delimiter='\n')
train_file = 'TrainLabels.txt'
test_file = 'TestLabels.txt'
f_train = open(train_file, 'w')
f_test = open(test_file, 'w')

for i, file in enumerate(partition): 
	part = file.split(' ')[1]
	if part is '0' or part is '1': 
		f_train.write(attribute[i + 2].replace('-1', '0') + '\n')
	else: 
		f_test.write(attribute[i + 2].replace('-1', '0') + '\n')
