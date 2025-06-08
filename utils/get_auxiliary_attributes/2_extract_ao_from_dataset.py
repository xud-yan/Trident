import torch

# 加载 .pth 文件
dataset = '{}' # please replace {} with the name of dataset (e.g., mit-states)
pth_file_path = '{}/metadata_compositional-split-natural.t7' # please replace {} with the root path of the dataset
d = torch.load(pth_file_path, map_location='cuda:0')
a = set()
o = set()
for data in d:
    a.add(data['attr'])
    o.add(data['obj'])

print(a)
with open(dataset +'_attr_name.txt', 'w') as f:
    for item in a:
        f.write(item + '\n')
print(o)
with open(dataset +'/_obj_name.txt', 'w') as f:
    for item in o:
        f.write(item + '\n')


with open(dataset +'/_word_name.txt', 'w') as f:
    for item in a:
        f.write(item + '\n')
    for item in o:
        f.write(item + '\n')