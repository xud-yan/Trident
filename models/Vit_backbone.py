import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig

class MM_projector(nn.Module): #cross-modal connector
    def __init__(self, cfg):
        super(MM_projector, self).__init__()

        self.fc1 = nn.Linear(1024, 4096)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4096, 4096)
        self.load(cfg.MODEL.Vit.mm_path, cfg.MODEL.device)
        self.to(cfg.MODEL.device)

    # @torch.no_grad()
    def forward(self, x):
        with torch.no_grad():
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            torch.cuda.empty_cache()
        return x

    def load(self, mm_path, map_location):
        state_dict = torch.load(mm_path, map_location)
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('0.'):
                new_key = key.replace('0.', 'fc1.')
            elif key.startswith('2.'):
                new_key = key.replace('2.', 'fc2.')
            else:
                new_key = key
            new_state_dict[new_key] = value

        self.load_state_dict(new_state_dict)


class CLIPVisionTower(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.vision_tower_path = cfg.MODEL.Vit.vision_tower_path
        self.select_layer = cfg.MODEL.Vit.mm_vision_select_layer
        self.select_feature = cfg.MODEL.Vit.mm_select_feature #local, global, all

        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_path, device_map=cfg.MODEL.Vit.device)
        self.vision_tower.requires_grad_(False)
        self.to(cfg.MODEL.Vit.device)
        self.device_str = cfg.MODEL.Vit.device

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        image_local_features = image_features[:, 1:]
        image_global_features = image_features[:, 0]
        if self.select_feature == 'local':
            return image_local_features, None
        elif self.select_feature == 'global':
            return None, image_global_features
        else:
            return image_local_features, image_global_features

    @torch.no_grad()
    def forward(self, images):
        images1 = images
        with torch.no_grad():
            image1_forward_outs = self.vision_tower(images1.to(device=self.device_str, dtype=self.dtype),
                                                    output_hidden_states=True)
        image1_local_features, image1_global_features = self.feature_select(image1_forward_outs)

        image1_local_features = image1_local_features.to(images.dtype)
        image1_global_features = image1_global_features.to(images.dtype)
        image1_local_features = image1_local_features.to(images.device)
        image1_global_features = image1_global_features.to(images.device)

        return image1_local_features, image1_global_features


    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device_str, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size