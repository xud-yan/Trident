import argparse
import torch
import torch.nn as nn
from models.trident import Trident
from dataset import CompositionDataset
import evaluator_ge
from tqdm import tqdm
from config import cfg


def validate_ge(model, testloader, evaluator, device, topks=[1, 2, 3]):
    model.eval()
    dset = testloader.dataset
    model.val_prepare(dset)

    _, _, all_attr_gt, all_obj_gt, all_pair_gt, all_pred = [], [], [], [], [], []
    for _, data in tqdm(enumerate(testloader), total=len(testloader), desc='Testing'):
        for k in data:
            if isinstance(data[k], list):
                continue
            data[k] = data[k].to(device, non_blocking=True)
        out = model(data)
        predictions = out['scores']
        attr_truth, obj_truth, pair_truth = data['attr'], data['obj'], data['pair']
        all_pred.append(predictions)
        all_attr_gt.append(attr_truth)
        all_obj_gt.append(obj_truth)
        all_pair_gt.append(pair_truth)

    all_attr_gt, all_obj_gt, all_pair_gt = torch.cat(all_attr_gt).to('cpu'), torch.cat(all_obj_gt).to(
        'cpu'), torch.cat(all_pair_gt).to('cpu')

    all_pred_dict = {}
    for k in all_pred[0].keys():
        all_pred_dict[k] = torch.cat(
            [all_pred[i][k].to('cpu') for i in range(len(all_pred))])

    # Calculate best unseen accuracy
    sel = ['AUC', 'closed_attr_match', 'closed_obj_match', 'best_unseen', 'best_seen', 'best_hm']
    for k in topks:
        results = evaluator.score_model(all_pred_dict, all_obj_gt, bias=1e3, topk=k)
        stats = evaluator.evaluate_predictions(results, all_attr_gt, all_obj_gt, all_pair_gt, all_pred_dict, topk=k)
        result = ''
        for key in stats:
            if key in sel:
                result = result + key + '  ' + str(round(stats[key], 4)) + '| '
        print(f'Top {k}')
        print(result)

    del model.val_attrs
    del model.val_objs

    return stats['AUC'], stats['best_hm']


parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, required=True, help='path to config file')
parser.add_argument('--load', type=str, required=True, help='path to model_file')
parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                    help='modify config file from terminal')
args = parser.parse_args()

cfg.merge_from_file(args.cfg)
cfg.merge_from_list(args.opts)
path_to_model = args.load
print(cfg)

# Prepare dataset & dataloader.
device = cfg.MODEL.device
print('Prepare dataset')
trainset = CompositionDataset(
        phase='train', split=cfg.DATASET.splitname, cfg=cfg)
valset = CompositionDataset(
            phase='val', split=cfg.DATASET.splitname, cfg=cfg)
valloader = torch.utils.data.DataLoader(
    valset, batch_size=cfg.TRAIN.test_batch_size, shuffle=False,
    num_workers=cfg.TRAIN.num_workers)
testset = CompositionDataset(
    phase='test', split=cfg.DATASET.splitname, cfg=cfg)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=cfg.TRAIN.test_batch_size, shuffle=False,
    num_workers=cfg.TRAIN.num_workers)

model = Trident(trainset, cfg)
model.to(device)
model.load_state_dict(torch.load(path_to_model), strict=False)
model.load_vit_backbone()

print(model)

evaluator_val_ge = evaluator_ge.Evaluator(valset, model)
evaluator_test_ge = evaluator_ge.Evaluator(testset, model)

topks = [1]
if cfg.DATASET.name == 'vaw-czsl':
    topks = [3, 5]

print('Val set:')
auc, best_hm = validate_ge(model, valloader, evaluator_val_ge, device, topks=topks)
print('Test set:')
auc, best_hm = validate_ge(model, testloader, evaluator_test_ge, device, topks=topks)
