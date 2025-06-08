attr_set = set()
dataset = '' # please replace {} with the name of the dataset
with open(dataset+'_attr_set.txt', 'r') as f:
    for line in f:
        attr_set.add(line.strip())

with open(dataset+'_aux_set.txt', 'r') as f:
    for line in f:
        attr_set.add(line.strip())
        
with open(dataset+'_obj_set.txt', 'r') as f:
    for line in f:
        attr_set.add(line.strip())

with open(dataset+'_word_set.txt', 'w') as f:
    for attr in attr_set:
        f.write(attr + '\n')