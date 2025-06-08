import copy

import numpy as np
import torch
import torch.utils.data as t_data
from transformers import CLIPImageProcessor

from PIL import Image
import json
from os.path import join as ospj

class ImageLoader:
    def __init__(self, root):
        self.img_dir = root

    def __call__(self, img):
        file = ospj(self.img_dir, img)
        img = Image.open(file).convert('RGB')
        return img


class CompositionDataset(t_data.Dataset):
    def __init__(
            self,
            phase,
            split='compositional-split',
            open_world=False,
            cfg=None
    ):
        self.phase = phase
        self.cfg = cfg
        self.split = split
        self.open_world = open_world

        self.vision_tower_path = cfg.MODEL.Vit.vision_tower_path
        self.loader = ImageLoader(ospj(cfg.DATASET.root_dir, 'images'))
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_path)

        self.attrs, self.objs, self.pairs, self.auxs, \
            self.train_pairs, self.val_pairs, \
            self.test_pairs, self.aux_by_pair = self.parse_split()

        self.words = self.attrs + self.objs
        for aux in self.auxs:
            if aux not in self.words:
                self.words.append(aux)

        self.train_data, self.val_data, self.test_data = self.get_split_info()
        if self.phase == 'train':
            self.data = self.train_data
        elif self.phase == 'val':
            self.data = self.val_data
        else:
            self.data = self.test_data

        self.obj2idx = {obj: idx for idx, obj in enumerate(self.objs)}
        self.attr2idx = {attr: idx for idx, attr in enumerate(self.attrs)}
        self.word2idx = {word: idx for idx, word in enumerate(self.words)}
        self.pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}

        self.train_pair2idx = {pair: idx for idx, pair in enumerate(self.train_pairs)}

        dic = {}
        for p in self.train_pairs:
            for item in self.aux_by_pair:
                key = list(item.keys())[0]
                if key == p:
                    aux_list = list(item.values())[0]
                    aux_list2ixd = [self.word2idx[word] for word in aux_list]
                    dic[self.train_pair2idx[p]] = aux_list2ixd
                    break

        self.aux2idx = dic  # Note!: aux is dependent on train data

        print('# train pairs: %d | # val pairs: %d | # test pairs: %d' % (len(
            self.train_pairs), len(self.val_pairs), len(self.test_pairs)))
        print('# train images: %d | # val images: %d | # test images: %d' %
              (len(self.train_data), len(self.val_data), len(self.test_data)))

        # Affordance.
        self.attr_affordance = {}  # -> contains objects compatible with an attribute.
        for _attr in self.attrs:
            candidates = [
                obj
                for (_, attr, obj) in self.train_data
                if attr == _attr
            ]
            self.attr_affordance[_attr] = sorted(list(set(candidates)))
            if len(self.attr_affordance[_attr]) <= 1:
                print(f'{_attr} is associated with <= 1 object: {self.attr_affordance[_attr]}')

        self.obj_affordance = {}  # -> contains objects compatible with an attribute.
        for _obj in self.objs:
            candidates = [
                attr
                for (_, attr, obj) in self.train_data
                if obj == _obj
            ]
            self.obj_affordance[_obj] = sorted(list(set(candidates)))
            if len(self.obj_affordance[_obj]) <= 1:
                print(f'{_obj} is associated with <= 1 object: {self.obj_affordance[_obj]}')

        # Images that contain an object.
        self.image_with_obj = {}
        for i, instance in enumerate(self.train_data):
            obj = instance[2]
            if obj not in self.image_with_obj:
                self.image_with_obj[obj] = []
            self.image_with_obj[obj].append(i)

        # Images that contain an attribute.
        self.image_with_attr = {}
        for i, instance in enumerate(self.train_data):
            attr = instance[1]
            if attr not in self.image_with_attr:
                self.image_with_attr[attr] = []
            self.image_with_attr[attr].append(i)

        # Images that contain a pair.
        self.image_with_pair = {}
        for i, instance in enumerate(self.train_data):
            attr, obj = instance[1], instance[2]
            if (attr, obj) not in self.image_with_pair:
                self.image_with_pair[(attr, obj)] = []
            self.image_with_pair[(attr, obj)].append(i)



    def get_split_info(self):
        data = torch.load(ospj(self.cfg.DATASET.root_dir, f'metadata_{self.split}.t7'))
        train_data, val_data, test_data = [], [], []

        for instance in data:
            image, attr, obj, settype = \
                instance['image'], instance['attr'], instance['obj'], instance['set']
            if attr == 'NA' or (attr, obj) not in self.pairs or settype == 'NA':
                continue
            data_i = [image, attr, obj]
            if settype == 'train':
                train_data.append(data_i)
            elif settype == 'val':
                val_data.append(data_i)
            else:
                test_data.append(data_i)

        return train_data, val_data, test_data

    def get_aux_by_pair(self, pairs, aux_list_file):
        aux_list = []
        aux_dic = {}
        with open(aux_list_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                aux_dic[list(data.keys())[0]] = list(data.values())[0]  # the string of pair

        for pair in pairs:
            aux_list.append({pair: aux_dic[pair[0] + ' ' + pair[1]]})
        return aux_list

    def parse_split(self):

        def parse_pairs(pair_list):
            with open(pair_list, 'r') as f:
                pairs = f.read().strip().split('\n')
                if self.cfg.DATASET.name.startswith('vaw'):
                    pairs = [t.split('+') for t in pairs]
                else:
                    pairs = [t.split() for t in pairs]
                pairs = list(map(tuple, pairs))
            attrs, objs = zip(*pairs)
            return attrs, objs, pairs

        tr_attrs, tr_objs, tr_pairs = parse_pairs(
            ospj(self.cfg.DATASET.root_dir, self.split, 'train_pairs.txt'))

        vl_attrs, vl_objs, vl_pairs = parse_pairs(
            ospj(self.cfg.DATASET.root_dir, self.split, 'val_pairs.txt'))

        ts_attrs, ts_objs, ts_pairs = parse_pairs(
            ospj(self.cfg.DATASET.root_dir, self.split, 'test_pairs.txt'))

        aux_by_pair = self.get_aux_by_pair(tr_pairs,
            ospj(self.cfg.DATASET.root_dir, self.cfg.DATASET.aux_name, 'aux_label.jsonl'))

        all_auxs = []

        for item in aux_by_pair:
            for key, value_list in item.items():  # only once
                for aux_word in value_list:
                    all_auxs.append(aux_word)

        all_attrs = sorted(list(set(tr_attrs + vl_attrs + ts_attrs))) #
        all_objs = sorted(list(set(tr_objs + vl_objs + ts_objs))) #
        all_auxs = sorted(list(set(all_auxs)))
        all_pairs = sorted(list(set(tr_pairs + vl_pairs + ts_pairs)))

        return all_attrs, all_objs, all_pairs, all_auxs,tr_pairs, vl_pairs, ts_pairs, aux_by_pair

    def image_preprocess(self, image):
        if self.cfg.DATASET.image_aspect_ratio == 'pad':
            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result

            image = expand2square(image, tuple(int(x * 255) for x in self.image_processor.image_mean))

        image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        return image

    def __getitem__(self, index):
        img, attr, obj = self.data[index]

        if self.phase == 'train':
            pair_idx = self.train_pair2idx[(attr, obj)]
            data = {
                    'attr': self.word2idx[attr],
                    'obj': self.word2idx[obj],
                    'pair': pair_idx,
                    'aux': torch.tensor(self.aux2idx[pair_idx]),
            }

            #data['mask_task'] = 1
            i2 = self.sample_same_attribute(attr, obj, with_different_obj=True)
            if i2 == -1: #i2 means the idx of the sample in train_data, instead of pair2idx or others
                data['attr_mask_task'] = 0
            else:
                data['attr_mask_task'] = 1

            img1_a, attr1, obj1_a = self.data[i2]
            pair_idx1_a = self.train_pair2idx[(attr1, obj1_a)]

            data['obj1_a'] = self.word2idx[obj1_a]
            data['pair1_a'] = pair_idx1_a
            data['aux1_a'] = torch.tensor(self.aux2idx[pair_idx1_a])

            i2 = self.sample_same_object(attr, obj, with_different_attr=True)
            if i2 == -1:
                data['obj_mask_task'] = 0
            else:
                data['obj_mask_task'] = 1

            img1_o, attr1_o, obj1 = self.data[i2]
            pair_idx1_o = self.train_pair2idx[(attr1_o, obj1)]

            data['attr1_o'] = self.word2idx[attr1_o]
            data['pair1_o'] = pair_idx1_o
            data['aux1_o'] = torch.tensor(self.aux2idx[pair_idx1_o])

            img = self.loader(img)
            img = self.image_preprocess(img)
            data['img'] = img

            img1_a = self.loader(img1_a)
            img1_a = self.image_preprocess(img1_a)
            data['img1_a'] = img1_a

            img1_o = self.loader(img1_o)
            img1_o = self.image_preprocess(img1_o)
            data['img1_o'] = img1_o

        else:
            # Testing mode.
            # ！！！test time： no word2idx，only attr2idx and obj2idx
            data = {
                'attr': self.attr2idx[attr],
                'obj': self.obj2idx[obj],
                'pair': self.pair2idx[(attr, obj)],
            }

            img = self.loader(img)
            img = self.image_preprocess(img)
            data['img'] = img

        return data

    def __len__(self):
        return len(self.data)

    def sample_same_attribute(self, attr, obj, with_different_obj=True):
        if with_different_obj:
            if len(self.attr_affordance[attr]) == 1:
                return -1
            i2 = np.random.choice(self.image_with_attr[attr])
            img1, attr1, obj1 = self.data[i2]
            while obj1 == obj:
                i2 = np.random.choice(self.image_with_attr[attr])
                img1, attr1, obj1 = self.data[i2]
            assert obj1 != obj
        else:
            i2 = np.random.choice(self.image_with_attr[attr])
        return i2

    def sample_same_object(self, attr, obj, with_different_attr=True):
        if with_different_attr:
            length = len(self.obj_affordance[obj])
            if length == 1:
                return -1

            i2 = np.random.choice(self.image_with_obj[obj])
            img1, attr1, obj1 = self.data[i2]
            while attr1 == attr:
                i2 = np.random.choice(self.image_with_obj[obj])
                img1, attr1, obj1 = self.data[i2]
            return i2
        else:
            i2 = np.random.choice(self.image_with_attr[attr])
        return i2

