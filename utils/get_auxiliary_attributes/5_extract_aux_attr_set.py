import json
attr_set = set()
dataset = '{}' # please replace {} with the name of the dataset
with open(dataset + '_aux_label.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        attr_list = list(data.values())[0]
        for attr in attr_list:
            attr_set.add(attr)


with open(dataset + '_aux_set.txt', 'w') as f:
    for attr in attr_set:
        f.write(attr + '\n')