import torch

# 加载 .pth 文件
pth_file_path = '{}/compositional-split-natural/' # please replace {} with the root path of the dataset
write_file_path = '{}_pair.txt' # please replace {} with the name of the dataset
p = set()

with open(f'{pth_file_path}train_pairs.txt', 'r') as f:
    for line in f:
        p.add(line[:-1])

with open(f'{pth_file_path}test_pairs.txt', 'r') as f:
    for line in f:
        p.add(line[:-1])

with open(f'{pth_file_path}val_pairs.txt', 'r') as f:
    for line in f:
        p.add(line[:-1])

with open(write_file_path, 'w') as f:
    for item in p:
        f.write(item + '\n')