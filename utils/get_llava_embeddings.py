# If you want to obtain the last hidden states of LLaVA v1.5
# please move this file to the root path of LLaVA v1.5 (!!!)
# and run this python script directly in LLaVA's environment
# LLaVA v1.5's code repository: https://github.com/haotian-liu/LLaVA

# After obtaining the file containing the last hidden states from LLaVA, please move it manually to your desired path.

#Note that we use the same ".txt" file format as GloVe and Word2Vec for storage to enhance code extensibility.

import sys
sys.path.append('')
import argparse
import torch

from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

def process(tokenizer, model, read_file, write_file, layer):
    embedding_list = []

    with open(read_file, 'r') as f:
        lines = f.readlines()
        words = [line.strip() for line in lines]

    for word in words:
        input_ids = [tokenizer(word).input_ids[1:]]
        input_ids = torch.tensor(input_ids, dtype=torch.long, device='cuda:0')
        print(input_ids)
        if input_ids.size(1) != 1:
            print(word + '------------------------------------------------------')
        with torch.inference_mode():
            output_embedding = model.forward(input_ids=input_ids, layer=layer)[0]
            print(output_embedding)
            output_embedding = torch.mean(output_embedding, dim=0)
            print(output_embedding)
            output_embedding.tolist()
        embedding_list.append(output_embedding)


    with open(write_file, 'w') as f:
        for i in range(len(words)):
            f.write(words[i] + ' ')
            for j in range(len(embedding_list[0])):
                f.write(f"{embedding_list[i][j]:.5f} ")
            f.write('\n')
    # After obtaining the file containing the last hidden states from LLaVA, please move it manually to your desired path.



def main(args):
    # Model
    disable_torch_init()
    dataset = "{}"  # please replace {} with the name of the dataset

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=args.model_path, model_base=args.model_base, model_name=model_name, device='cuda:0'
    )
    # the following read_file_list is obtained by Trident/utils/get_aux/2_extract_ao_from_dataset.py
    # please copy these {dataset}_word_name.txt files and this python script to the root path of LLaVA v1.5 (!!!)
    read_file_list = [dataset + '_word_name.txt']
    write_file_list = [dataset + '_llava_32_embedding.txt']
    layer = [32] # 32 means the last hidden states
    for i in range(len(read_file_list)):
        for j in range(len(layer)):
            process(tokenizer, model, read_file_list[i], write_file_list[i*2+j], layer[j])



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="{}") # please replace {} with the root path of LLaVA-v1.5-7B
    parser.add_argument("--model-base", type=str, default=None)
    args = parser.parse_args()

    main(args)
