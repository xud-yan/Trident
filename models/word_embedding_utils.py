import torch
import numpy as np
import re


def load_word_embeddings(emb_file, vocab):
    model = {}

    for line in open(emb_file, 'r'):
        line = line.strip().split(' ')
        wvec = torch.FloatTensor(list(map(float, line[1:])))
        model[line[0]] = wvec

    # for zappos (should account for everything)
    custom_map = {
        'Faux.Fur':'fake_fur', 'Faux.Leather':'fake_leather', 'Full.grain.leather':'thick_leather', 
        'Hair.Calf':'hair_leather', 'Patent.Leather':'shiny_leather', 'Nubuck':'grainy_leather', 
        'Boots.Ankle':'ankle_boots', 'Boots.Knee.High':'knee_high_boots', 'Boots.Mid-Calf':'midcalf_boots', 
        'Shoes.Boat.Shoes':'boat_shoes', 'Shoes.Clogs.and.Mules':'clogs_shoes', 'Shoes.Flats':'flats_shoes',
        'Shoes.Heels':'heels', 'Shoes.Loafers':'loafers', 'Shoes.Oxfords':'oxford_shoes',
        'Shoes.Sneakers.and.Athletic.Shoes':'sneakers'}
    custom_map_vaw = {
        'selfie': 'photo'
    }

    embeds = []
    for k in vocab:
        if k in custom_map:
            k = custom_map[k]
        if '_' in k or '-' in k or k not in model:
            ks = re.split('[_-]', k)
            for id, tmp_k in enumerate(ks):
                if tmp_k in custom_map_vaw:
                    ks[id] = custom_map_vaw[tmp_k]
            try:
                emb = np.stack([model[it] for it in ks if it in model]).mean(axis=0)
            except Exception as e:
                if 'llava' in emb_file:
                    emb = torch.zeros(4096).float()
        else:
            emb = model[k]
        embeds.append(emb)
    embeds = torch.Tensor(np.stack(embeds))
    print('Embeddings loaded, total embeddings: {}'.format(embeds.size()))
    return embeds


def initialize_wordembedding_matrix(name, vocab, cfg):
    """
    Args:
    - name: hyphen separated word embedding names: 'glove-word2vec-conceptnet'.
    - vocab: list of attributes/objects.
    """

    if 'llava' in name:
        wordemb_ = load_word_embeddings(
            cfg.MODEL.llava_embedding_path, vocab)

    if 'llava' in name:
        dim = 4096
    return wordemb_, dim
