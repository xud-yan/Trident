import argparse
import numpy as np
import os
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from bisect import bisect_right
from models.trident import Trident
from dataset import CompositionDataset
import evaluator_ge
from tqdm import tqdm
from utils import utils
from config import cfg
from torch.utils.tensorboard import SummaryWriter
import wandb
from datetime import datetime

def freeze(m):
    """Freezes module m.
    """
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
        p.grad = None

def decay_learning_rate_milestones(group_lrs, optimizer, epoch, cfg):
    """Decays learning rate following milestones in cfg.
    """
    milestones = cfg.TRAIN.lr_decay_milestones
    it = bisect_right(milestones, epoch)
    gamma = cfg.TRAIN.decay_factor ** it
    
    gammas = [gamma] * len(group_lrs)
    assert len(optimizer.param_groups) == len(group_lrs)
    i = 0
    for param_group, lr, gamma_group in zip(optimizer.param_groups, group_lrs, gammas):
        param_group["lr"] = lr * gamma_group
        i += 1
        print(f"Group {i}, lr = {lr * gamma_group}")


def save_checkpoint(model_or_optim, name, cfg):
    """Saves checkpoint.
    """
    state_dict = model_or_optim.state_dict()

    keys_to_remove = [k for k in state_dict.keys() if k.startswith('feat_extractor') ] # or k.startswith('mm_projector')]
    for key in keys_to_remove:
        del state_dict[key]

    path = os.path.join(
        f'{cfg.TRAIN.checkpoint_dir}/{cfg.config_name}_{cfg.TRAIN.seed}/{name}.pth')
    torch.save(state_dict, path)


def train(epoch, model, optimizer, trainloader, wandb_dict, device, cfg):
    model.train()

    list_meters = ['loss_total', 'pair_loss']
    if cfg.MODEL.use_dis_loss:
        list_meters.append('dis_loss')
    if cfg.MODEL.use_orthogonal_regularization_loss:
        list_meters.append('ortho_loss')

    dict_meters = { 
        k: utils.AverageMeter() for k in list_meters
    }

    acc_attr_meter = utils.AverageMeter()
    acc_obj_meter = utils.AverageMeter()
    acc_pair_meter = utils.AverageMeter()

    out = None
    for idx, batch in enumerate(tqdm(trainloader)):

        for k in batch:
            batch[k] = batch[k].to(device, non_blocking=True)
        
        out = model(batch)

        loss = out['loss_total']

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if 'acc_attr' in out:
            acc_attr_meter.update(out['acc_attr'])
            acc_obj_meter.update(out['acc_obj'])
        acc_pair_meter.update(out['acc_pair'])
        for k in out:
            if k in dict_meters:
                dict_meters[k].update(out[k].item())

    # 下面是记录
    print(
        f'Epoch: {epoch} Iter: {idx+1}/{len(trainloader)}, '
        f'Loss: {dict_meters["loss_total"].avg:.3f}, '
        f'Acc_Pair: {acc_pair_meter.avg:.2f}, ',
        flush=True)

    wandb_dict['loss'] = dict_meters["loss_total"].avg


    wandb_dict['train/acc_attr'] = acc_attr_meter.avg
    wandb_dict['train/acc_obj'] = acc_obj_meter.avg
    wandb_dict['train/acc_pair'] = acc_pair_meter.avg

    acc_pair_meter.reset()
    if 'acc_attr' in out:
        acc_attr_meter.reset()
        acc_obj_meter.reset()
    for k in out:
        if k in dict_meters:
            dict_meters[k].reset()

def validate_ge(epoch, model, testloader, evaluator, device, topk=1):
    model.eval()

    dset = testloader.dataset
    model.val_prepare(dset)

    _, _, all_attr_gt, all_obj_gt, all_pair_gt, all_pred = [], [], [], [], [], []
    for _, data in tqdm(enumerate(testloader), total=len(testloader), desc='Testing'):
        for k in data:
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
    # Gather values as dict of (attr, obj) as key and list of predictions as values
    for k in all_pred[0].keys():
        all_pred_dict[k] = torch.cat(
            [all_pred[i][k].to('cpu') for i in range(len(all_pred))])

    # Calculate best unseen accuracy
    results = evaluator.score_model(all_pred_dict, all_obj_gt, bias=1e3, topk=topk)
    stats = evaluator.evaluate_predictions(results, all_attr_gt, all_obj_gt, all_pair_gt, all_pred_dict, topk=topk)

    stats['a_epoch'] = epoch

    result = ''
    # write to Tensorboard
    for key in stats:
        result = result + key + '  ' + str(round(stats[key], 4)) + '| '

    print(f'Val Epoch: {epoch}')
    print(result)

    return stats['AUC'], stats['best_hm']


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main_worker(cfg):
    """Main training code.
    """
    seed = cfg.TRAIN.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = cfg.MODEL.device
    print(f'Use GPU {device} for training', flush=True)
    #torch.cuda.set_device(gpu)

    # Directory to save checkpoints.
    ckpt_dir = f'{cfg.TRAIN.checkpoint_dir}/{cfg.config_name}_{cfg.TRAIN.seed}'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    
    print('Prepare dataset')
    trainset = CompositionDataset(
        phase='train', split=cfg.DATASET.splitname, cfg=cfg)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=cfg.TRAIN.batch_size, shuffle=True,
        num_workers=cfg.TRAIN.num_workers,
        pin_memory=True, drop_last=False, worker_init_fn=seed_worker)

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
    start_epoch = cfg.TRAIN.start_epoch

    model.load_vit_backbone()

    print(model)

    freeze(model.feat_extractor)

    evaluator_val_ge = evaluator_ge.Evaluator(valset, model)
    evaluator_test_ge = evaluator_ge.Evaluator(testset, model)


    params_word_embedding = []
    params_encoder = []
    params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if 'attr_embedder' in name or 'obj_embedder' in name or 'word_embedder' in name or 'mm_projector' in name:
            if cfg.TRAIN.lr_word_embedding > 0:
                params_word_embedding.append(p)
                print('params_word_embedding: %s' % name)
        elif name.startswith('feat_extractor'):
            params_encoder.append(p)
            print('params_encoder: %s' % name)
        else:
            params.append(p)
            print('params_main: %s' % name)

    optimizer = optim.Adam([
        {'params': params_word_embedding, 'lr': cfg.TRAIN.lr_word_embedding},
        {'params': params, 'lr': cfg.TRAIN.lr},
    ], lr=cfg.TRAIN.lr, weight_decay=cfg.TRAIN.wd)
    group_lrs = [cfg.TRAIN.lr_word_embedding, cfg.TRAIN.lr]

    epoch = start_epoch

    val_best_auc = 0.0
    test_best_auc = 0.0

    while epoch <= cfg.TRAIN.max_epoch:
        wandb_dict = {}

        train(epoch, model, optimizer, trainloader, wandb_dict, device, cfg)

        if cfg.TRAIN.decay_strategy == 'milestone':
            decay_learning_rate_milestones(group_lrs, optimizer, epoch, cfg)

        if epoch < cfg.TRAIN.start_epoch_validate:
            pass
        elif epoch % cfg.TRAIN.eval_every_epoch == 0:
            # Validate.
            print('Validation set ===>')
            val_auc, val_hm = validate_ge(epoch, model, valloader, evaluator_val_ge, device, topk=cfg.EVAL.topk)

            wandb_dict['val/auc'] = val_auc
            wandb_dict['val/best_hm'] = val_hm

            if val_auc > val_best_auc:
                val_best_auc = val_auc

            print('Test set ===>')
            # Test.
            test_auc, test_hm = validate_ge(epoch, model, testloader, evaluator_test_ge, device, topk=cfg.EVAL.topk)

            wandb_dict['test/auc'] = test_auc
            wandb_dict['test/best_hm'] = test_hm

            if test_auc > test_best_auc:
                test_best_auc = test_best_auc

            save_checkpoint(model, f'model_epoch{epoch}', cfg)
            print(f'save model_epoch{epoch}')

        if cfg.TRAIN.use_wandb:
            wandb_dict['epoch'] = epoch
            wandb.log(wandb_dict)
        epoch += 1
    
    print('Done: %s' % cfg.config_name)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, help='path to config file')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='modify config file from terminal')
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    if cfg.TRAIN.use_wandb:
        wandb.init(project='Trident-' + cfg.config_name, config={"start_epoch": cfg.TRAIN.start_epoch})

    # 获取当前时间
    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
    print("Beginning Time：", formatted_now)

    print(cfg)

    seed = cfg.TRAIN.seed
    print('Random seed:', seed)
    cfg.TRAIN.seed = seed

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    
    main_worker(cfg)

    if cfg.TRAIN.use_wandb:
        wandb.finish()
    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
    print("Ending Time：", formatted_now)


if __name__ == "__main__":
    main()