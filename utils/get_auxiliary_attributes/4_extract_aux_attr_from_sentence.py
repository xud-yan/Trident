import json
import re
import shutil
import os

dataset = '{}' # please replace {} with the name of the dataset
dataset_dir = '{}' # please replace {} with the root path of the dataset
with open(dataset + '_aux_label.jsonl', 'w') as wf:
    with open(dataset + '_aux_init_content.jsonl', 'r') as rf:
        for line in rf:
            data = json.loads(line)
            pair_gt = list(data.keys())[0]
            attr_gt = pair_gt.split(' ')[0]
            sentence = list(data.values())[0].lower()
            words = re.split(r'[^a-zA-Z-]+', sentence)
            words = [word for word in words if word]
            if len(words) != 6:
                print(pair_gt, '***', len(words), '***', sentence)
            if words[0] == attr_gt:
                words[0] = words[3]
            if words[1] == attr_gt:
                words[1] = words[3]
            if words[2] == attr_gt:
                words[2] = words[3]
            dic = {pair_gt: words[:3]}
            json.dump(dic, wf)
            wf.write('\n')

# copy the auxiliary attribute files to the path of dataset
os.makedirs(dataset_dir + '/aux_label/', exist_ok=True)
dst_file = os.path.join(dataset_dir + '/aux_label/', 'aux_label.jsonl')
shutil.copy(dataset + '_aux_label.jsonl', dst_file)