import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Vit_backbone import CLIPVisionTower, MM_projector
from .basic_layers import MLP, MLP_For_Remain
from .word_embedding_utils import initialize_wordembedding_matrix





class Trident(nn.Module):
    """Object-Attribute Compositional Learning from Image Pair.
    """

    def __init__(self, dset, cfg):
        super(Trident, self).__init__()
        self.cfg = cfg
        self.device = cfg.MODEL.device

        self.num_attrs = len(dset.attrs)
        self.num_objs = len(dset.objs)
        self.num_words = len(dset.words)
        self.pair2idx = dset.pair2idx

        # Set training pairs.
        train_attrs_by_pair, train_objs_by_pair = zip(*dset.train_pairs)
        train_attrs_by_pair = [dset.word2idx[attr] for attr in train_attrs_by_pair]
        train_objs_by_pair = [dset.word2idx[obj] for obj in train_objs_by_pair]
        train_words = dset.word2idx
        self.train_attrs_by_pair = torch.LongTensor(train_attrs_by_pair).to(cfg.MODEL.device) # without aux
        self.train_objs_by_pair = torch.LongTensor(train_objs_by_pair).to(cfg.MODEL.device)
        self.words = train_words

        self.emb_dim = cfg.MODEL.emb_dim

        # Setup layers for word embedding composer.
        self._setup_word_composer(dset, cfg)

        adaptive_aggregation_modules = [
            nn.Conv1d(in_channels=4096, out_channels=cfg.MODEL.img_emb_dim, kernel_size=1),
            nn.BatchNorm1d(cfg.MODEL.img_emb_dim),
            nn.ReLU()
        ]
        feat_dim = cfg.MODEL.img_emb_dim

        if cfg.MODEL.img_emb_drop > 0:
            adaptive_aggregation_modules += [
                nn.Dropout1d(cfg.MODEL.img_emb_drop)]

        self.faa = nn.Sequential(*adaptive_aggregation_modules)
        self.image_embedder = nn.Linear(feat_dim * 2, self.emb_dim)


        self.classifier = CosineClassifier(temp=cfg.MODEL.cosine_cls_temp)

        self.global_feature_num = cfg.MODEL.global_feature_num
        self.local_feature_num = cfg.MODEL.local_feature_num
        self.global_cond_masks = nn.Embedding(self.global_feature_num, self.emb_dim)
        self.local_agg_module = []
        for i in range(self.local_feature_num):
            conv = nn.Sequential(
                nn.Conv1d(in_channels=self.emb_dim, out_channels=1, kernel_size=1),
                nn.Sigmoid()
            )
            conv.to(cfg.MODEL.device)
            self.local_agg_module.append(conv)

        self.ao_dis = AO_DIS_MODULE(
            cfg, self.num_words, self.word_embedder,
            self.word_dim, feat_dim
        )

        self.mm_projector = MM_projector(self.cfg)
        if self.emb_dim != 1024:
            self.global_pre = MLP(
                1024, self.cfg.MODEL.emb_dim, self.cfg.MODEL.emb_dim, 2, batchnorm=False,
                drop_input=cfg.MODEL.wordemb_compose_dropout
            )


    def load_vit_backbone(self):
        self.feat_extractor = CLIPVisionTower(self.cfg)  # Backbone('resnet18')


    def _setup_word_composer(self, dset, cfg):

        word_wordemb, self.word_dim = \
            initialize_wordembedding_matrix(cfg.MODEL.wordembs, dset.words, cfg)

        self.word_embedder = nn.Embedding(self.num_words, self.word_dim)
        self.word_embedder.weight.data.copy_(word_wordemb)


        self.wordemb_compose = cfg.MODEL.wordemb_compose

        self.compose = nn.Sequential(
            nn.Dropout(cfg.MODEL.wordemb_compose_dropout),
            nn.Linear(self.word_dim * 2, self.emb_dim)
        )

    def compose_word_embeddings(self, mode='train'):
        if mode == 'train': # train
            attr_emb = self.word_embedder(self.train_attrs_by_pair)  # [n_pairs, word_dim].
            obj_emb = self.word_embedder(self.train_objs_by_pair)  # # [n_pairs, word_dim].
        elif mode == 'all':
            attr_emb = self.word_embedder(self.all_attrs)  # [n_pairs, word_dim].
            obj_emb = self.word_embedder(self.all_objs)
        elif mode == 'unseen':
            attr_emb = self.word_embedder(self.unseen_pair_attrs)  # [n_pairs, word_dim].
            obj_emb = self.word_embedder(self.unseen_pair_objs)
        else: # test
            attr_embedder = nn.Embedding(self.num_attrs, self.word_dim)
            obj_embedder = nn.Embedding(self.num_objs, self.word_dim)
            attr_embedder.to(self.device)
            obj_embedder.to(self.device)
            attr_embedder.weight.data.copy_(self.word_embedder.weight.data[:self.num_attrs])
            obj_embedder.weight.data.copy_(self.word_embedder.weight.data[self.num_attrs: self.num_attrs + self.num_objs])

            attr_emb = attr_embedder(self.val_attrs)  # [n_pairs, word_dim].
            obj_emb = obj_embedder(self.val_objs)  # # [n_pairs, word_dim].


        concept_emb = torch.cat((obj_emb, attr_emb), dim=-1)
        concept_emb = self.compose(concept_emb)  # [n_pairs, emb_dim].

        return concept_emb

    def get_global_features(self, features):
        global_features = []
        for idx in range(self.global_feature_num):
            concept_idx = np.zeros((features.size(0),), dtype=int)
            concept_idx += idx
            concept_idx = torch.from_numpy(concept_idx)
            concept_idx = concept_idx.to(self.device)
            concept_idx = torch.autograd.Variable(concept_idx)
            mask = self.global_cond_masks(concept_idx)  # batch size * dim
            mask = nn.functional.relu(mask)
            if self.emb_dim != 1024:
                features_g = self.global_pre(features)
                global_features.append(features_g * mask)
            else:
                global_features.append(features * mask)

        global_features = torch.stack(global_features).permute(1, 0, 2).contiguous()
        return global_features

    def get_local_features(self, features):
        local_features = []
        for i in range(self.local_feature_num):
            torch.cuda.empty_cache()
            weight = self.local_agg_module[i](features.mT)
            feature_i = torch.mean(features.mT * weight, dim=-1)
            local_features.append(feature_i)

        local_features = torch.stack(local_features).permute(1, 0, 2).contiguous()
        torch.cuda.empty_cache()
        return local_features

    def normal_cross_entropy(self, pred, target):
        output = torch.tensor(0.0, dtype=pred.dtype, device=pred.device)

        for i, data_line in enumerate(pred):  # length是数据量
            deno = torch.exp(data_line).sum()  #deno means denominator
            output += -pred[i][target[i]] + torch.log(deno)
        return output / len(target)

    def compute_loss_with_normal_ce(self, f, concept, labels):
        pred = self.classifier(f, concept)
        emb_pair_loss = self.normal_cross_entropy(pred, labels)
        return emb_pair_loss

    def train_forward(self, batch):

        attr_labels = batch['attr']
        obj_labels = batch['obj']
        pair_labels = batch['pair']
        aux_labels = batch['aux']

        img2_a_pair_labels = batch['pair1_a']
        obj2_a_labels = batch['obj1_a']
        aux2_a_labels = batch['aux1_a']
        attr_mask_task = batch['attr_mask_task']

        img2_o_pair_labels = batch['pair1_o']
        attr2_o_labels = batch['attr1_o']
        aux2_o_labels = batch['aux1_o']
        obj_mask_task = batch['obj_mask_task']

        bs = self.cfg.TRAIN.batch_size

        concept = self.compose_word_embeddings(mode='train')  # (n_pairs, emb_dim)

        img1 = batch['img']
        img2_a = batch['img1_a']  # Image that shares the same attribute
        img2_o = batch['img1_o']  # Image that shares the same object
        img1_local, img1_global = self.feat_extractor(img1)
        img2_a_local, img2_a_global = self.feat_extractor(img2_a)
        img2_o_local, img2_o_global = self.feat_extractor(img2_o)


        img1_local = self.mm_projector(img1_local)
        img2_a_local = self.mm_projector(img2_a_local)
        img2_o_local = self.mm_projector(img2_o_local)

        img1_local = self.faa(img1_local.mT).mT
        img2_a_local = self.faa(img2_a_local.mT).mT
        img2_o_local = self.faa(img2_o_local.mT).mT

        img1_local = self.get_local_features(img1_local)  # batchsize * num(8) * dim
        img2_a_local = self.get_local_features(img2_a_local)
        img2_o_local = self.get_local_features(img2_o_local)

        img1_global = self.get_global_features(img1_global)
        img2_a_global = self.get_global_features(img2_a_global)
        img2_o_global = self.get_global_features(img2_o_global)

        img1_local_mean = torch.mean(img1_local, dim=1)  # The first dim 0 is batch size
        img2_a_local_mean = torch.mean(img2_a_local, dim=1)
        img2_o_local_mean = torch.mean(img2_o_local, dim=1)

        img1_global_mean = torch.mean(img1_global, dim=1)
        img2_a_global_mean = torch.mean(img2_a_global, dim=1)
        img2_o_global_mean = torch.mean(img2_o_global, dim=1)

        img1_l_g_mean = torch.cat((img1_local_mean, img1_global_mean), dim=-1)
        img2_a_l_g_mean = torch.cat((img2_a_local_mean, img2_a_global_mean), dim=-1)
        img2_o_l_g_mean = torch.cat((img2_o_local_mean, img2_o_global_mean), dim=-1)

        del img1_local_mean, img1_global_mean
        del img2_a_local_mean, img2_a_global_mean, img2_o_local_mean, img2_o_global_mean

        img1_l_g_mean = self.image_embedder(img1_l_g_mean)
        img2_a_l_g_mean = self.image_embedder(img2_a_l_g_mean)
        img2_o_l_g_mean = self.image_embedder(img2_o_l_g_mean)


        pair_loss = (
                    self.compute_loss_with_normal_ce(img1_l_g_mean, concept, pair_labels) +
                    self.compute_loss_with_normal_ce(img2_a_l_g_mean, concept, img2_a_pair_labels) +
                    self.compute_loss_with_normal_ce(img2_o_l_g_mean, concept, img2_o_pair_labels)
                    ) / 3.0


        loss = pair_loss

        pred = self.classifier(img1_l_g_mean, concept)
        pred = torch.max(pred, dim=1)[1]
        attr_pred = self.train_attrs_by_pair[pred]
        obj_pred = self.train_objs_by_pair[pred]

        correct_attr = (attr_pred == attr_labels)
        correct_obj = (obj_pred == obj_labels)
        correct_pair = (pred == pair_labels)

        out = {
            'acc_attr': torch.div(correct_attr.sum(), float(bs)),
            'acc_obj': torch.div(correct_obj.sum(), float(bs)),
            'acc_pair': torch.div(correct_pair.sum(), float(bs)),
            'pair_loss': pair_loss
        }

        if self.cfg.MODEL.use_orthogonal_regularization_loss:
            ortho_loss = (
                         self.orthogonal_regularization(img1_global) +
                         self.orthogonal_regularization(img1_local) +
                         self.orthogonal_regularization(img2_a_global) +
                         self.orthogonal_regularization(img2_a_local) +
                         self.orthogonal_regularization(img2_o_global) +
                         self.orthogonal_regularization(img2_o_local)
                         ) / 6.0
            loss = loss + self.cfg.MODEL.w_loss_ortho * ortho_loss
            out['ortho_loss'] = ortho_loss

        img1_l_g = torch.cat([img1_local, img1_global], dim=1)
        img2_a_l_g = torch.cat([img2_a_local, img2_a_global], dim=1)
        img2_o_l_g = torch.cat([img2_o_local, img2_o_global], dim=1)

        if self.cfg.MODEL.use_dis_loss:
            aux_loss = self.ao_dis(
                img1_l_g, img2_a_l_g, img2_o_l_g, attr_labels, obj_labels, attr2_o_labels, obj2_a_labels,
                attr_mask_task, obj_mask_task, aux_labels, aux2_a_labels, aux2_o_labels
            )
            loss = loss + aux_loss['dis_loss'] * self.cfg.MODEL.w_loss_dis
            out['dis_loss'] = aux_loss['dis_loss']
            out['aux_acc'] = aux_loss['acc']

        out['loss_total'] = loss


        return out

    def val_prepare(self, dset):
        val_attrs, val_objs = zip(*dset.pairs)
        val_attrs = [dset.attr2idx[attr] for attr in val_attrs]
        val_objs = [dset.obj2idx[obj] for obj in val_objs]
        self.val_attrs = torch.LongTensor(val_attrs).to(self.device)
        self.val_objs = torch.LongTensor(val_objs).to(self.device)
        self.val_pairs = dset.pairs

    def val_forward(self, batch):
        bs = self.cfg.TRAIN.batch_size

        concept = self.compose_word_embeddings(mode='val')  # [n_pairs, emb_dim].

        img = batch['img']
        img_local, img_global = self.feat_extractor(img)

        img_local = self.mm_projector(img_local)
        img_local = self.faa(img_local.mT).mT
        img_local = self.get_local_features(img_local)
        img_local_mean = torch.mean(img_local, dim=1)  # The first dim is batch size

        img_global = self.get_global_features(img_global)
        img_global_mean = torch.mean(img_global, dim=1)

        img = torch.cat((img_local_mean, img_global_mean), dim=-1)
        img = self.image_embedder(img)

        pred = self.classifier(img, concept, scale=False)

        out = {}
        out['pred'] = pred

        out['scores'] = {}
        for _, pair in enumerate(self.val_pairs):
            out['scores'][pair] = pred[:, self.pair2idx[pair]]

        return out


    def forward(self, x):
        if self.training:
            out = self.train_forward(x)
        else:
            with torch.no_grad():
                out = self.val_forward(x)
        return out

    def orthogonal_regularization(self, templates):
        # batch_size, length, dim
        batch_size, length, dim = templates.size()
        device = templates.device
        norm_templates = F.normalize(templates, p=2, dim=-1)
        # (B,L,D) * (B,D,L)
        cosine_score = torch.matmul(norm_templates,
                                    norm_templates.permute(0, 2, 1).contiguous())  # batch_size, length, length
        eye_matrix = torch.eye(length).unsqueeze(0).repeat(batch_size, 1, 1).to(device)
        l2_loss = torch.nn.MSELoss()
        return l2_loss(cosine_score, eye_matrix)



class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, input1, input2):
        cosine_similarity = F.cosine_similarity(input1, input2, dim=1)
        loss = 1 - cosine_similarity.mean()
        return loss


class AO_DIS_MODULE(nn.Module):
    """Cross attention module to find difference/similarity between two images.
    """

    def __init__(
            self,
            cfg,
            num_words,
            word_embedder,
            word_dim,
            emb_dim
    ):
        super(AO_DIS_MODULE, self).__init__()
        self.cfg = cfg

        self.num_words = num_words
        self.words = torch.LongTensor(list(range(self.num_words))).to(cfg.MODEL.device)

        self.word_embedder = word_embedder

        self.shared_weights_1 = MLP_For_Remain(emb_dim)
        self.shared_weights_2 = MLP_For_Remain(emb_dim)
        self.ao_2_word = nn.Linear(emb_dim * 2, emb_dim)

        self.word_mlp = MLP(inp_dim=word_dim, latent_dim=word_dim // 2, out_dim=emb_dim,
                            dropout=cfg.MODEL.wordemb_compose_dropout, end_relu=False)

        self.label_smoothing_alpha = cfg.MODEL.label_smoothing_alpha

        self.classify_word = CosineClassifier(cfg.MODEL.cosine_cls_temp)

    def forward_attn(self, image1, image2, fg1=None, fg2=None):
        image_both = torch.cat([image1, image2], dim=-1)
        shared_weight_1 = self.shared_weights_1(image_both)
        diff_weight_1 = 1 - shared_weight_1

        shared_weight_2 = self.shared_weights_2(image_both)
        diff_weight_2 = 1 - shared_weight_2

        shared_image_1 = torch.mean(shared_weight_1 * image1, dim=1)
        shared_image_2 = torch.mean(shared_weight_2 * image2, dim=1)

        diff_image_1 = torch.mean(diff_weight_1 * image1, dim=1)
        diff_image_2 = torch.mean(diff_weight_2 * image2, dim=1)

        shared_image = self.ao_2_word(torch.cat([shared_image_1, shared_image_2], dim=-1))

        return shared_image, shared_image_1, shared_image_2, diff_image_1, diff_image_2

    def normal_cross_entropy(self, pred, target):
        output = torch.tensor(0.0, dtype=pred.dtype, device=pred.device)

        for i, data_line in enumerate(pred):  # length是数据量
            deno = torch.exp(data_line).sum()
            output += -pred[i][target[i]] + torch.log(deno)
        return output / len(target)

    def label_smoothing_cross_entropy(self, pred, target, aux):
        alpha = self.label_smoothing_alpha
        output = torch.tensor(0.0, dtype=pred.dtype, device=pred.device)

        single_alpha = alpha / aux.size(-1)

        for i, data_line in enumerate(pred):
            deno = torch.exp(data_line).sum()
            output += -(1 - alpha) * pred[i][target[i]] + torch.log(deno)
            for aux_idx in aux[i]:
                output += - single_alpha * pred[i][aux_idx]
        return output / len(target)

    def compute_loss_with_normal_ce(self, features, word_embed, label):
        pred = self.classify_word(features, word_embed)
        loss = self.normal_cross_entropy(pred, label)
        pred = torch.max(pred, dim=1)[1]
        pred = self.words[pred]
        correct = (pred == label)
        acc = torch.div(correct.sum().float(), label.size(0))
        return loss, acc

    def compute_loss_with_ls_ce(self, features, word_embed, label, aux):
        pred = self.classify_word(features, word_embed)
        loss = self.label_smoothing_cross_entropy(pred, label, aux)
        pred = torch.max(pred, dim=1)[1]
        pred = self.words[pred]
        correct = (pred == label)
        acc = torch.div(correct.sum().float(), label.size(0))
        return loss, acc

    def forward(self, img1, img2_a, img2_o, attr1, obj1, attr_2_o, obj_2_a, attr_mask_task, obj_mask_task,
                aux_labels, aux2_a_labels, aux2_o_labels):
        """

        """
        shared_image, shared_image_1, shared_image_2, diff_image_1, diff_image_2_a = self.forward_attn(img1, img2_a)
        # ↑ the common part is attribute, the difference part is object, such as blue bus and blue car
        shared_image_, shared_image_1_, shared_image_2_, diff_image_1_, diff_image_2_o_ = self.forward_attn(img1, img2_o)
        # ↑ the common part is object, the difference part is attribute, such as blue bus and red bus

        word_emb = self.word_embedder(self.words)
        word_emb = self.word_mlp(word_emb)
        attr_mask = (attr_mask_task == 1)
        obj_mask = (obj_mask_task == 1)

        out = {
            'attr_mask': attr_mask,
            'obj_mask': obj_mask,
            'shared_image': shared_image[attr_mask],
            'diff_image_1': diff_image_1[attr_mask],
            'diff_image_2_a': diff_image_2_a[attr_mask],
            'shared_image_': shared_image_[obj_mask],
            'diff_image_1_': diff_image_1_[obj_mask],
            'diff_image_2_o_': diff_image_2_o_[obj_mask]
        }

        loss1, acc1 = self.compute_loss_with_normal_ce(shared_image[attr_mask], word_emb, attr1[attr_mask]) # 3 column blue of blue bus and blue car
        loss2, acc2 = self.compute_loss_with_ls_ce(shared_image_1[attr_mask], word_emb, attr1[attr_mask], aux_labels[attr_mask]) # 2 column blue of blue bus
        loss3, acc3 = self.compute_loss_with_ls_ce(shared_image_2[attr_mask], word_emb, attr1[attr_mask], aux2_a_labels[attr_mask]) # 2 column blue of blue car
        loss4, acc4 = self.compute_loss_with_normal_ce(diff_image_1[attr_mask], word_emb, obj1[attr_mask]) # 2 column diff bus of blue bus
        loss5, acc5 = self.compute_loss_with_normal_ce(diff_image_2_a[attr_mask], word_emb, obj_2_a[attr_mask]) # 2 column diff car of blue car

        loss6, acc6 = self.compute_loss_with_normal_ce(shared_image_[obj_mask], word_emb, obj1[obj_mask]) # 3 column bus of blue bus and red bus
        loss7, acc7 = self.compute_loss_with_normal_ce(shared_image_1_[obj_mask], word_emb, obj1[obj_mask]) # 2 column bus of blue bus
        loss8, acc8 = self.compute_loss_with_normal_ce(shared_image_2_[obj_mask], word_emb, obj1[obj_mask]) # 2 column bus of red bus
        loss9, acc9 = self.compute_loss_with_ls_ce(diff_image_1_[obj_mask], word_emb, attr1[obj_mask], aux_labels[obj_mask]) # 2 column diff blue of blue bus
        loss10, acc10 = self.compute_loss_with_ls_ce(diff_image_2_o_[obj_mask], word_emb, attr_2_o[obj_mask], aux2_o_labels[obj_mask]) #2 column diff red of red bus

        out['dis_loss'] = (loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8 + loss9 + loss10) / 10.0
        out['acc'] = (acc1 + acc2 + acc3 + acc4 + acc5 + acc6) / 6.0

        return out


class CosineClassifier(nn.Module):
    def __init__(self, temp=0.05):
        super(CosineClassifier, self).__init__()
        self.temp = temp

    def forward(self, img, concept, scale=True):
        """
        img: (bs, emb_dim)
        concept: (n_class, emb_dim)
        """
        img_norm = F.normalize(img, dim=-1)
        concept_norm = F.normalize(concept, dim=-1)
        pred = torch.matmul(img_norm, concept_norm.transpose(0, 1))
        if scale:
            pred = pred / self.temp
        return pred
