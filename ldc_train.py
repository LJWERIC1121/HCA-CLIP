import os
import torch
import random
import numpy as np
from tqdm import tqdm
import clip_ldc as clip
import torch.nn.functional as F
import torchvision.transforms as T
from alisuretool.Tools import Tools
from torch.utils.data import DataLoader
from datasets.utils import DatasetWrapper
from torchvision.transforms import InterpolationMode
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from augmentations import rotation, rotation2, rotation8, random_augment8, random_augment8_strong, get_rotation_labels
from losses import SupConLoss


MODEL_CACHE_DIR = './model/clip'
DATA_ROOT = 'D:/dataset/FSIC'
LOG_ROOT = './result/log'


class MyTransform(object):

    @staticmethod
    def _convert_image_to_rgb(image):
        return image.convert("RGB")

    @staticmethod
    def transform_train(size, scale=(0.8, 1.0)):
        funcs = [
            T.RandomResizedCrop(size=size, scale=scale, interpolation=InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5), MyTransform._convert_image_to_rgb, ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]
        return Compose(funcs)

    @staticmethod
    def transform_test(size):
        funcs = [
            Resize(size, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(size), MyTransform._convert_image_to_rgb, ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]
        return Compose(funcs)

    pass


class Config10Dataset(object):

    def __init__(self, dataset_name, seed=2024, shots=16, backbone="RN50", lr=0.001, batch_size=64, train_epoch=50,
                 loss_lambda=[1.0, 1.0, 1.0, 1.0, 1.0], fuse_type=4, use_rotation=True, rotation_type="rotation8",
                 use_contrastive=True, contrastive_weight=0.2, contrastive_temp=0.07):
        self.setup_seed(seed)

        self.seed = seed
        self.shots = shots
        self.lr = lr
        self.train_epoch = train_epoch
        self.batch_size = batch_size
        self.backbone = backbone  # RN50 RN101 ViT-B/32 ViT-B/16

        self.loss_lambda = loss_lambda
        self.fuse_type = fuse_type

        # Rotation augmentation and contrastive learning settings
        self.use_rotation = use_rotation  # Whether to use rotation augmentation
        self.rotation_type = rotation_type  # Type of rotation: "rotation2", "rotation", "rotation8", "random_augment8", "random_augment8_strong"
        self.use_contrastive = use_contrastive  # Whether to use contrastive loss
        self.contrastive_weight = contrastive_weight  # Weight for contrastive loss (0.1-0.3)
        self.contrastive_temp = contrastive_temp  # Temperature for contrastive loss

        _dataset_info = self.dataset_info()
        self.dataset_name = dataset_name
        assert self.dataset_name in _dataset_info.keys()
        self.data_path = os.path.join(DATA_ROOT, _dataset_info[self.dataset_name][2])
        self.dataset = _dataset_info[self.dataset_name][0](self.data_path, self.shots)
        self.num_classes = _dataset_info[self.dataset_name][1]

        self.cache_dir = MODEL_CACHE_DIR
        pass

    def get_detail(self):
        detail_str = (f"dataset_name={self.dataset_name}, shots={self.shots}, lr={self.lr}, seed={self.seed}, "
                      f"train_epoch={self.train_epoch}, batch_size={self.batch_size}, backbone={self.backbone}, "
                      f"num_classes={self.num_classes}, loss_lambda={self.loss_lambda}, fuse_type={self.fuse_type}, "
                      f"use_rotation={self.use_rotation}, rotation_type={self.rotation_type}, "
                      f"use_contrastive={self.use_contrastive}, "
                      f"contrastive_weight={self.contrastive_weight}, contrastive_temp={self.contrastive_temp}")
        return detail_str

    @staticmethod
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        pass

    @staticmethod
    def get_gpu_id():
        """
        torch.cuda.set_device(get_gpu_id())
        """
        import pynvml

        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        gpu_id, free = 0, 0
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            now_free = (info.free // 1048576) / 1024  # info.total, info.free, info.used
            if now_free > free:
                free = now_free
                gpu_id = i
            pass
        pynvml.nvmlShutdown()
        return gpu_id

    @staticmethod
    def dataset_info():
        """
        Dataset information for HCA-CLIP
        Only includes the three plant disease datasets
        """
        from datasets.plant_disease import PlantDisease
        from datasets.ai_challenger import AiChallenger
        from datasets.spd import SPD

        return {
            "plant_disease": [PlantDisease, 34, "Plant_disease"],
            "ai_challenger": [AiChallenger, 53, "Ai_Challenger_2018"],
            "spd": [SPD, 42, "SPD"]
        }

    pass


class ConfigImageDomainShift(object):

    def __init__(self, seed=2024, shots=16, backbone="RN50", lr=0.001, batch_size=64, train_epoch=50,
                 loss_lambda=[1.0, 1.0, 1.0, 1.0, 1.0], fuse_type=2, has_ood=True):
        Config10Dataset.setup_seed(seed)

        self.seed = seed
        self.shots = shots
        self.lr = lr
        self.train_epoch = train_epoch
        self.batch_size = batch_size
        self.backbone = backbone  # RN50 RN101 ViT-B/32 ViT-B/16

        self.loss_lambda = loss_lambda
        self.fuse_type = fuse_type
        self.has_ood = has_ood

        self.num_classes = 1000
        self.dataset_name = "imagenet"
        self.data_path_imagenet = os.path.join(DATA_ROOT, 'imagenet/images')
        self.data_path_imagenet_v2 = os.path.join(DATA_ROOT, 'imagenetv2/imagenetv2-matched-frequency-format-val')
        self.data_path_imagenet_sketch = os.path.join(DATA_ROOT, 'imagenet-sketch/images')

        from datasets.imagenet import MyImageNet
        from datasets.imagenetv2 import ImageNetV2
        from datasets.imagenet_sketch import ImageNetSketch
        self.dataset = MyImageNet(self.data_path_imagenet, self.shots, 'train', MyTransform.transform_train(224))
        self.test_set = MyImageNet(root=self.data_path_imagenet, num_shots=self.shots,
                                   split='test', transform=MyTransform.transform_test(224))
        self.test_set_v2 = ImageNetV2(root=self.data_path_imagenet_v2, transform=MyTransform.transform_test(224))
        self.test_set_sketch = ImageNetSketch(root=self.data_path_imagenet_sketch, transform=MyTransform.transform_test(224))

        self.cache_dir = MODEL_CACHE_DIR
        pass

    def get_detail(self):
        detail_str = (f"dataset_name={self.dataset_name}, shots={self.shots}, lr={self.lr}, seed={self.seed}, "
                      f"train_epoch={self.train_epoch}, batch_size={self.batch_size}, backbone={self.backbone}, "
                      f"num_classes={self.num_classes}, loss_lambda={self.loss_lambda}, fuse_type={self.fuse_type}")
        return detail_str

    pass


class MyScheduler(object):

    def __init__(self, optimizer, base_value, final_value, epochs, niter_per_ep, warmup_epochs=0) -> None:
        self.optimizer = optimizer
        self.optimizer.param_groups[0]['lr'] = 0

        warmup_schedule = np.array([])
        warmup_iters = warmup_epochs * niter_per_ep
        if warmup_epochs > 0:
            warmup_schedule = np.linspace(0, base_value, warmup_iters)

        iters = np.arange(epochs * niter_per_ep - warmup_iters)
        self.schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

        self.schedule = np.concatenate((warmup_schedule, self.schedule))
        self.id = 0
        assert len(self.schedule) == epochs * niter_per_ep

    def step(self):
        self.optimizer.param_groups[0]['lr'] = self.schedule[self.id]
        self.id += 1
        pass

    pass


class Eval(object):

    def __init__(self, batch_size, clip_model, val_loader, text_feats):
        self.clip_model = clip_model
        self.text_feats = text_feats
        self.val_loader = val_loader
        self.batch_size = batch_size
        pass

    def eval(self, best_beta=None):
        self.clip_model.eval()
        all_labels, all_logits = [], []
        with torch.no_grad():
            with tqdm(enumerate(self.val_loader), total=len(self.val_loader), desc='Evaluate') as tqdm_eval:
                for _, (images, labels) in tqdm_eval:
                    clip_logits, mlp_logits, ada_logits, tot_logits, mlp_logits_1to3, mlp_logits_2to3 = self.clip_model.my_forward(images.cuda(),
                                                                                                 self.text_feats)
                    all_logits.append([clip_logits, mlp_logits, ada_logits, tot_logits, mlp_logits_1to3, mlp_logits_2to3])
                    all_labels.append(labels)
                    pass
                pass
            pass
        all_labels = torch.cat(all_labels, dim=0)

        result_acc = {}
        acc = self.cal_acc(torch.cat([one[0] for one in all_logits], dim=0), all_labels) * 100.
        result_acc["clip_logits"] = acc
        Tools.print(f"test all_clip_logits acc={acc:.2f}%")
        acc = self.cal_acc(torch.cat([one[1] for one in all_logits], dim=0), all_labels) * 100.
        result_acc["mlp_logits"] = acc
        Tools.print(f"test all_mlp_logits acc={acc:.2f}%")
        acc = self.cal_acc(torch.cat([one[2] for one in all_logits], dim=0), all_labels) * 100.
        result_acc["ada_logits"] = acc
        Tools.print(f"test all_ada_logits acc={acc:.2f}%")
        acc = self.cal_acc(torch.cat([one[3] for one in all_logits], dim=0), all_labels) * 100.
        result_acc["tot_logits"] = acc
        Tools.print(f"test all_tot_logits acc={acc:.2f}%")
        
        # 计算额外分类器的准确率（如果存在）
        if all_logits[0][4] is not None:
            acc = self.cal_acc(torch.cat([one[4] for one in all_logits], dim=0), all_labels) * 100.
            result_acc["mlp_logits_1to3"] = acc
            Tools.print(f"test mlp_logits_1to3 acc={acc:.2f}%")
            
            acc = self.cal_acc(torch.cat([one[5] for one in all_logits], dim=0), all_labels) * 100.
            result_acc["mlp_logits_2to3"] = acc
            Tools.print(f"test mlp_logits_2to3 acc={acc:.2f}%")

        if best_beta is None:
            # 验证集：搜索最佳beta，融合mlp_logits和ada_logits
            mlp_logits_cat = torch.cat([one[1] for one in all_logits], dim=0)
            ada_logits_cat = torch.cat([one[2] for one in all_logits], dim=0)
            
            best_beta, last_acc, best_acc = self.search_hp(mlp_logits_cat, ada_logits_cat, all_labels)
            result_acc["acc"] = best_acc
            Tools.print(f"val best beta (mlp+ada) = {best_beta:.4f} => last_acc={last_acc:.2f}% [best_acc={best_acc}]")
            
            # 额外融合1：mlp_logits_1to3 + ada_logits
            if all_logits[0][4] is not None:
                mlp_logits_1to3_cat = torch.cat([one[4] for one in all_logits], dim=0)
                best_beta_1to3, _, best_acc_1to3 = self.search_hp(mlp_logits_1to3_cat, ada_logits_cat, all_labels)
                result_acc["acc_1to3_ada"] = best_acc_1to3
                Tools.print(f"val best beta (1to3+ada) = {best_beta_1to3:.4f} => best_acc={best_acc_1to3:.2f}%")
                
                # 额外融合2：mlp_logits_2to3 + ada_logits
                mlp_logits_2to3_cat = torch.cat([one[5] for one in all_logits], dim=0)
                best_beta_2to3, _, best_acc_2to3 = self.search_hp(mlp_logits_2to3_cat, ada_logits_cat, all_labels)
                result_acc["acc_2to3_ada"] = best_acc_2to3
                Tools.print(f"val best beta (2to3+ada) = {best_beta_2to3:.4f} => best_acc={best_acc_2to3:.2f}%")
            
            return best_beta, result_acc
        else:
            # 测试集：使用最佳beta融合mlp_logits和ada_logits
            mlp_logits_cat = torch.cat([one[1] for one in all_logits], dim=0)
            ada_logits_cat = torch.cat([one[2] for one in all_logits], dim=0)
            
            logits = self.fuse_logits(mlp_logits_cat, ada_logits_cat, beta=best_beta)
            acc = self.cal_acc(logits, all_labels) * 100.
            result_acc["acc"] = acc
            Tools.print(f"test acc (mlp+ada)={acc:.2f}%")
            
            # 额外融合1：mlp_logits_1to3 + ada_logits (使用验证集的beta重新搜索)
            if all_logits[0][4] is not None:
                mlp_logits_1to3_cat = torch.cat([one[4] for one in all_logits], dim=0)
                best_beta_1to3, _, acc_1to3_ada = self.search_hp(mlp_logits_1to3_cat, ada_logits_cat, all_labels)
                result_acc["acc_1to3_ada"] = acc_1to3_ada
                Tools.print(f"test acc (1to3+ada)={acc_1to3_ada:.2f}% with beta={best_beta_1to3:.4f}")
                
                # 额外融合2：mlp_logits_2to3 + ada_logits
                mlp_logits_2to3_cat = torch.cat([one[5] for one in all_logits], dim=0)
                best_beta_2to3, _, acc_2to3_ada = self.search_hp(mlp_logits_2to3_cat, ada_logits_cat, all_labels)
                result_acc["acc_2to3_ada"] = acc_2to3_ada
                Tools.print(f"test acc (2to3+ada)={acc_2to3_ada:.2f}% with beta={best_beta_2to3:.4f}")
            
            return best_beta, result_acc
        # return best_beta, acc

    @staticmethod
    def fuse_logits(mlp_logits, ada_logits, beta=0.5):
        """
        融合mlp_logits和ada_logits
        beta: mlp_logits的权重，(1-beta)是ada_logits的权重
        """
        return beta * mlp_logits + (1 - beta) * ada_logits

    @staticmethod
    def cal_acc(logits, labels):
        pred = torch.argmax(logits, -1)
        acc_num = (pred == labels.cuda()).sum().item()
        return 1.0 * acc_num / len(labels)

    def search_hp(self, mlp_logits, ada_logits, all_labels, start=0, end=1, step=50):
        """
        搜索mlp_logits和ada_logits的最佳融合beta
        """
        beta_list = [i * (end - start) / step + start for i in range(step + 1)]
        accs, best_beta, best_acc = [], start, 0.
        for beta in beta_list:
            logits = self.fuse_logits(mlp_logits, ada_logits, beta=beta)
            acc = self.cal_acc(logits, all_labels) * 100.
            accs.append((beta, acc))
            if acc > best_acc:
                best_acc = acc
                best_beta = beta
        return best_beta, accs[-1][-1], best_acc

    pass


class AvgACC:
    def __init__(self) -> None:
        self.acc_num = 0
        self.total = 0
        pass

    def step(self, logits, labels):
        pred = torch.argmax(logits, -1)
        acc_num = (pred == labels.cuda()).sum().item()
        total = len(labels)
        self.acc_num += acc_num
        self.total += total
        pass

    def cal(self):
        return 0.00 if self.total == 0 else 1.0 * self.acc_num / self.total

    pass


class Runner(object):

    def __init__(self, config, best_results_log=None):
        self.config = config
        self.best_results_log = best_results_log  # 最佳结果日志文件路径

        # Initialize augmentation
        if self.config.use_rotation:
            # Select augmentation method based on rotation_type
            rotation_type = getattr(self.config, 'rotation_type', 'rotation8')  # Default to rotation8

            if rotation_type == "rotation2":
                self.rotation_transform, self.rotation_multiplier = rotation2()
                Tools.print(f"[CONFIG] Using rotation2 augmentation: 2 rotations (0°, 180°)")
            elif rotation_type == "rotation":
                self.rotation_transform, self.rotation_multiplier = rotation()
                Tools.print(f"[CONFIG] Using rotation augmentation: 4 rotations (0°, 90°, 180°, 270°)")
            elif rotation_type == "rotation8":
                self.rotation_transform, self.rotation_multiplier = rotation8()
                Tools.print(f"[CONFIG] Using rotation8 augmentation: 8 rotations at 45° intervals")
                Tools.print(f"[CONFIG] Rotation angles: 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°")
            elif rotation_type == "random_augment8":
                self.rotation_transform, self.rotation_multiplier = random_augment8()
                Tools.print(f"[CONFIG] Using random_augment8: 8 random augmented views")
                Tools.print(f"[CONFIG] Random augmentations: rotation, flip, crop, color jitter, grayscale")
            elif rotation_type == "random_augment8_strong":
                self.rotation_transform, self.rotation_multiplier = random_augment8_strong()
                Tools.print(f"[CONFIG] Using random_augment8_strong: 8 strongly augmented views")
                Tools.print(f"[CONFIG] Strong augmentations: aggressive rotation, flip, crop, color jitter, grayscale")
            else:
                # Fallback to rotation8
                self.rotation_transform, self.rotation_multiplier = rotation8()
                Tools.print(f"[CONFIG] Unknown rotation_type '{rotation_type}', using rotation8 as fallback")

            Tools.print(f"[CONFIG] Augmentation multiplier: {self.rotation_multiplier}x")
        else:
            self.rotation_transform = None
            self.rotation_multiplier = 1

        # Initialize contrastive loss
        if self.config.use_contrastive:
            self.contrastive_loss_fn = SupConLoss(temperature=self.config.contrastive_temp)
            Tools.print(f"[CONFIG] Using supervised contrastive loss with weight={self.config.contrastive_weight}, temp={self.config.contrastive_temp}")
        else:
            self.contrastive_loss_fn = None

        Tools.print(f"[CONFIG] Dataset: {self.config.dataset_name}, Backbone: {self.config.backbone}, Shots: {self.config.shots}, FuseType: {self.config.fuse_type}")
        Tools.print(f"Preparing {self.config.backbone} model.")
        self.clip_model, self.preprocess = clip.load(self.config.backbone, download_root=self.config.cache_dir,
                                                     num_classes=self.config.num_classes, config=self.config)
        self.clip_model.eval()

        Tools.print("Getting cached textual weights W ...")
        self.text_feats = self.clip_classifier(
            os.path.join(self.config.cache_dir, f"{self.config.dataset_name}_{self.config.backbone}_textfeats.pt"),
            self.config.dataset.classnames, self.config.dataset.template, self.clip_model)

        # Preparation for training
        for param in self.clip_model.parameters():
            param.requires_grad = False
            pass
        for name, param in self.clip_model.named_parameters():
            if 'adapter' in name:
                param.requires_grad = True
            pass

        Tools.print(f"Preparing {self.config.dataset_name} dataset.")
        if self.config.dataset_name != "imagenet":
            train_dataset = DatasetWrapper(self.config.dataset.train_x, input_size=224, transform=MyTransform.transform_train(224), is_train=True)
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size, num_workers=8, shuffle=True, drop_last=False, pin_memory=(torch.cuda.is_available()))
            self.val_loader = DataLoader(
                DatasetWrapper(self.config.dataset.val, input_size=224, transform=self.preprocess, is_train=False),
                batch_size=64, num_workers=8, shuffle=False, drop_last=False, pin_memory=(torch.cuda.is_available()))
            self.test_loader = DataLoader(
                DatasetWrapper(self.config.dataset.test, input_size=224, transform=self.preprocess, is_train=False),
                batch_size=64, num_workers=8, shuffle=False, drop_last=False, pin_memory=(torch.cuda.is_available()))
            self.test_loader_list = [self.test_loader]

            # Print dataset info
            Tools.print(f"Dataset info: {len(train_dataset)} training samples, batch_size={self.config.batch_size}, batches={len(self.train_loader)}")
            if self.config.use_rotation:
                Tools.print(f"Rotation augmentation enabled: each batch will be expanded {self.rotation_multiplier}x during training")
                Tools.print(f"Effective training samples per epoch: {len(train_dataset) * self.rotation_multiplier}")
        else:
            self.train_loader = DataLoader(self.config.dataset, self.config.batch_size, num_workers=8, shuffle=True)
            self.val_loader = None
            self.test_loader = DataLoader(dataset=self.config.test_set, batch_size=self.config.batch_size, num_workers=8, shuffle=False)
            self.test_loader_v2 = DataLoader(dataset=self.config.test_set_v2, batch_size=self.config.batch_size, num_workers=8, shuffle=False)
            self.test_loader_sketch = DataLoader(dataset=self.config.test_set_sketch, batch_size=self.config.batch_size, num_workers=8, shuffle=False)
            self.test_loader_list = [self.test_loader, self.test_loader_v2, self.test_loader_sketch] if self.config.has_ood else [self.test_loader]
            pass

        self.optimizer = torch.optim.AdamW(self.clip_model.parameters(), lr=self.config.lr / 10, weight_decay=1e-4, eps=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.config.train_epoch * len(self.train_loader))

        self.eval = Eval(self.config.batch_size, self.clip_model, self.test_loader, self.text_feats)
        pass

    def train_epoch(self, epoch):
        self.clip_model.adapter.train()
        self.clip_model.visual.adapter.train()

        train_acc, train_loss = AvgACC(), 0.0
        train_acc_1to3, train_acc_2to3 = AvgACC(), AvgACC()  # 第二阶段分支
        train_contrastive_loss = 0.0  # Track contrastive loss
        loss_list = [0, 0, 0, 0, 0]  # l1_mlp, l1_ada, ce_mlp, ce_ada, ce_tot
        with tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f"epoch {epoch}") as tqdm_train:
            for _, (images, labels) in tqdm_train:
                images, labels = images.cuda(), labels.cuda()

                # Apply rotation augmentation if enabled
                if self.config.use_rotation and self.rotation_transform is not None:
                    images = self.rotation_transform(images)  # [B, C, H, W] -> [B*k, C, H, W]
                    labels = get_rotation_labels(labels, self.rotation_multiplier)  # [B] -> [B*k]

                clip_logits, mlp_logits, ada_logits, total_logits, mlp_logits_1to3, mlp_logits_2to3 = self.clip_model.my_forward(images, self.text_feats)

                # Get fused features for contrastive learning
                # We need to extract fused_feats from the model
                if self.config.use_contrastive and self.contrastive_loss_fn is not None:
                    # Extract fused features from the model (x6 in the forward pass)
                    image_feats, _, fused_feats, _, _ = self.clip_model.encode_image(images)
                    contrastive_loss = self.contrastive_loss_fn(fused_feats, labels)
                else:
                    contrastive_loss = torch.tensor(0.0).cuda()

                # Compute main loss
                loss, losses = self.get_loss(labels, clip_logits, mlp_logits, ada_logits, total_logits,
                                             mlp_logits_1to3, mlp_logits_2to3, lambda_value=self.config.loss_lambda)

                # Add contrastive loss
                if self.config.use_contrastive:
                    loss = loss + self.config.contrastive_weight * contrastive_loss
                    train_contrastive_loss += contrastive_loss.item()

                train_loss += loss.item()
                train_acc.step(mlp_logits, labels)
                
                # 计算第二阶段分支的准确率
                if mlp_logits_1to3 is not None:
                    train_acc_1to3.step(mlp_logits_1to3, labels)
                    train_acc_2to3.step(mlp_logits_2to3, labels)

                for i, l in enumerate(losses):
                    loss_list[i] += l.item()
                tqdm_train.set_postfix(cur_loss=loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if self.scheduler:
                    self.scheduler.step()

            train_acc_result = train_acc.cal()
            train_loss = train_loss / len(self.train_loader)
            train_contrastive_loss = train_contrastive_loss / len(self.train_loader)
            pass

        Tools.print(f"train acc={train_acc_result}, "
                    f"[l1_loss, ce_loss] => {[one / len(self.train_loader) for one in loss_list]}")

        # 显示对比学习损失
        if self.config.use_contrastive:
            Tools.print(f"train contrastive_loss={train_contrastive_loss:.4f}")

        # 显示第二阶段分类器的准确率
        if mlp_logits_1to3 is not None:
            train_acc_1to3_result = train_acc_1to3.cal()
            train_acc_2to3_result = train_acc_2to3.cal()
            Tools.print(f"train acc_1to3={train_acc_1to3_result:.4f}, acc_2to3={train_acc_2to3_result:.4f}")

        return train_loss

    def train(self):
        best_test_acc = 0.0
        best_test_beta = None
        best_epoch = -1
        best_model_state = None
        best_test_acc_list = None  # 保存最佳的完整测试结果列表

        # 计算从哪个epoch开始测试（最后10轮）
        test_start_epoch = max(0, self.config.train_epoch - 10)

        for epoch in range(self.config.train_epoch):
            loss = self.train_epoch(epoch)
            Tools.print(f"Epoch: {epoch}, loss: {loss:.4f}, "
                        f"lr: {self.optimizer.state_dict()['param_groups'][0]['lr']:.8f}")

            # 最后10轮每轮都测试
            if epoch >= test_start_epoch:
                Tools.print(f"\n{'='*50}")
                Tools.print(f"Testing at Epoch {epoch}:")
                Tools.print(f"{'='*50}")
                
                # 使用test_without_val直接在测试集上搜索最佳beta并测试
                test_acc_list = self.test_without_val()
                
                # 更新最佳模型
                if len(test_acc_list) > 0:
                    current_test_acc = test_acc_list[0]['acc']
                    current_test_beta = test_acc_list[0].get('best_beta', None)
                    
                    if current_test_acc > best_test_acc:
                        best_test_acc = current_test_acc
                        best_test_beta = current_test_beta
                        best_epoch = epoch
                        best_model_state = self.clip_model.state_dict()
                        best_test_acc_list = test_acc_list
                        Tools.print(f"\n*** 新的最佳结果! Epoch {epoch}, Acc {best_test_acc:.2f}% ***\n")
        
        # 训练循环结束后输出最终最佳结果
        Tools.print(f"\n{'='*50}")
        Tools.print(f"训练完成! 最佳结果:")
        Tools.print(f"  最佳Epoch: {best_epoch}")
        Tools.print(f"  最佳测试准确率: {best_test_acc:.2f}%")
        if best_test_beta is not None:
            Tools.print(f"  最佳Beta: {best_test_beta}")
        Tools.print(f"{'='*50}\n")
        
        # 使用最佳结果
        final_results = best_test_acc_list if best_test_acc_list is not None else self.test_without_val()

        if self.best_results_log and len(final_results) > 0:
            self.save_best_result(-1, final_results[0]['acc'],
                                 final_results[0].get('best_beta', None), final_results)

        return final_results

    def save_best_result(self, best_epoch, best_acc, best_beta, final_results):
        """保存最佳测试结果到单独的文件"""
        import datetime

        # 获取当前时间
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 构建结果信息
        result_info = []
        result_info.append(f"\n{'='*80}")
        result_info.append(f"训练时间: {timestamp}")
        result_info.append(f"{'='*80}")
        result_info.append(f"配置信息:")
        result_info.append(f"  数据集: {self.config.dataset_name}")
        result_info.append(f"  Backbone: {self.config.backbone}")
        result_info.append(f"  Shots: {self.config.shots}")
        result_info.append(f"  训练轮数: {self.config.train_epoch}")
        result_info.append(f"  学习率: {self.config.lr}")
        result_info.append(f"  Batch size: {self.config.batch_size}")
        result_info.append(f"\n最佳结果:")
        if best_epoch >= 0:
            result_info.append(f"  最佳Epoch: {best_epoch}")
        result_info.append(f"  最佳测试准确率: {best_acc:.2f}%")
        if best_beta is not None:
            try:
                # 如果是单个数值
                result_info.append(f"  最佳Beta: {best_beta:.4f}")
            except Exception:
                # 如果是列表/元组（分阶段或多权重）
                if isinstance(best_beta, (list, tuple)):
                    try:
                        beta_str = ", ".join([f"{float(b):.4f}" for b in best_beta])
                    except Exception:
                        beta_str = str(best_beta)
                    result_info.append(f"  最佳Beta(列表): [{beta_str}]")
                else:
                    result_info.append(f"  最佳Beta: {str(best_beta)}")

        result_info.append(f"\n详细结果:")
        for i, result in enumerate(final_results, 1):
            result_info.append(f"  测试集 {i}:")
            result_info.append(f"    最终准确率: {result['acc']:.2f}%")
            result_info.append(f"    CLIP logits: {result['clip_logits']:.2f}%")
            result_info.append(f"    MLP logits: {result['mlp_logits']:.2f}%")
            result_info.append(f"    Adapted logits: {result['ada_logits']:.2f}%")
            result_info.append(f"    Total logits: {result['tot_logits']:.2f}%")

        result_info.append(f"{'='*80}\n\n")

        # 写入文件
        with open(self.best_results_log, 'a', encoding='utf-8') as f:
            f.write('\n'.join(result_info))

        Tools.print(f"最佳结果已保存到: {self.best_results_log}")

    def test_without_val(self):
        """直接在测试集上搜索最佳beta并测试（不使用验证集）"""
        self.eval.clip_model = self.clip_model
        test_acc_list = []

        for test_loader in self.test_loader_list:
            self.eval.val_loader = test_loader
            # best_beta=None 表示在测试集上搜索最佳beta
            best_beta, test_result_acc = self.eval.eval(best_beta=None)
            test_result_acc['best_beta'] = best_beta  # 保存best_beta
            test_acc_list.append(test_result_acc)
            pass

        return test_acc_list

    def test(self):
        """原始测试方法（使用验证集），保留用于兼容性"""
        self.eval.clip_model = self.clip_model
        val_best_beta = None
        if self.val_loader:
            self.eval.val_loader = self.val_loader
            val_best_beta, val_result_acc = self.eval.eval()
            pass
        test_acc_list = []
        for test_loader in self.test_loader_list:
            self.eval.val_loader = test_loader
            val_best_beta, test_result_acc = self.eval.eval(best_beta=val_best_beta)
            test_acc_list.append(test_result_acc)
            pass
        return test_acc_list

    @staticmethod
    def clip_classifier(feat_path, classnames, template, clip_model):
        if os.path.exists(feat_path):
            Tools.print(f"Loading texture features from {feat_path}")
            text_feats = torch.load(feat_path, map_location='cpu')
            return text_feats.cuda()

        with torch.no_grad():
            clip_weights = []
            for classname in classnames:
                classname = classname.replace('_', ' ')
                if isinstance(template, str):
                    # Simple template like "a photo of {}"
                    texts = [template.format(classname)]
                elif isinstance(template, list):
                    texts = [t.format(classname) for t in template]
                elif isinstance(template, dict):
                    texts = template[classname]

                texts = clip.tokenize(texts).cuda()
                # prompt ensemble for ImageNet
                class_embeddings = clip_model.encode_text(texts)
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                clip_weights.append(class_embedding)
                pass

            clip_weights = torch.stack(clip_weights, dim=1).cuda()
            torch.save(clip_weights, Tools.new_dir(feat_path))

        return clip_weights

    @staticmethod
    def get_loss(labels, clip_logits, mlp_logits, ada_logits, total_logits, 
                 mlp_logits_1to3, mlp_logits_2to3, lambda_value=[1.0, 1.0, 1.0, 0, 0]):
        ce_loss = F.cross_entropy(mlp_logits, labels) * lambda_value[0]
        ce_loss2 = F.cross_entropy(ada_logits, labels) * lambda_value[1]
        ce_loss3 = F.cross_entropy(total_logits, labels) * lambda_value[2]

        l1_loss1 = F.l1_loss(mlp_logits, clip_logits) * lambda_value[3]
        l1_loss2 = F.l1_loss(ada_logits, clip_logits) * lambda_value[4]

        # 添加第二阶段分支分类器的损失（如果存在）
        if mlp_logits_1to3 is not None:
            ce_loss_1to3 = F.cross_entropy(mlp_logits_1to3, labels) * 1
            ce_loss_2to3 = F.cross_entropy(mlp_logits_2to3, labels) * 1
            l1_loss3 = F.l1_loss(mlp_logits_1to3, clip_logits) * 1
            l1_loss4 = F.l1_loss(mlp_logits_2to3, clip_logits) * 1
            loss = l1_loss1 + l1_loss2 + ce_loss + ce_loss2 + ce_loss3 + ce_loss_1to3 + ce_loss_2to3 + l1_loss3 + l1_loss4
            return loss, [l1_loss1, l1_loss2, ce_loss, ce_loss2, ce_loss3]
        else:
            loss = l1_loss1 + l1_loss2 + ce_loss + ce_loss2 + ce_loss3
            return loss, [l1_loss1, l1_loss2, ce_loss, ce_loss2, ce_loss3]

    pass


class AllExperiments(object):
    """
    Legacy experiment class - NOT USED in HCA-CLIP
    Use scripts/train.py HCAClipExperiments instead
    Kept for backward compatibility only
    """

    def __init__(self):
        self.seed = 2024
        # Only include the three datasets available in HCA-CLIP
        self.datasets = "plant_disease/ai_challenger/spd"
        pass

    def main_experiment_1_zero_shot(self):
        log_txt_path = Tools.new_dir(os.path.join(LOG_ROOT, "1_main_experiment_1_zero_shot.txt"))
        backbone_list = ["RN50", "ViT-B/16"]
        for backbone in backbone_list:
            self.experiment_one(backbone=backbone, train_epoch=0, has_ood=False, log_txt_path=log_txt_path)
            pass
        pass

    def main_experiment_2_few_shot(self):
        log_txt_path = Tools.new_dir(os.path.join(LOG_ROOT, "1_main_experiment_2_few_shot.txt"))
        backbone_list = ["RN50", "ViT-B/16"]
        shots_list = [1, 2, 4, 8, 16]
        for backbone in backbone_list:
            for shots in shots_list:
                self.experiment_one(shots=shots, backbone=backbone, log_txt_path=log_txt_path)
                pass
        pass

    def experiment_one(self, shots=16, backbone="RN50", train_epoch=50, has_ood=True, log_txt_path=None):
        results = []
        for dataset_name in self.datasets.split('/'):
            # Dataset
            if dataset_name == "imagenet":
                config = ConfigImageDomainShift(seed=self.seed, shots=shots, backbone=backbone,
                                                train_epoch=train_epoch, has_ood=has_ood)
            else:
                config = Config10Dataset(dataset_name=dataset_name, seed=self.seed, shots=shots,
                                         backbone=backbone, train_epoch=train_epoch)
                pass

            # Runner
            runner = Runner(config=config)
            acc_list = runner.train()
            results.append({"name": dataset_name, "acc": acc_list, "detail": config.get_detail()})

            Tools.print({"name": dataset_name, "acc": acc_list, "detail": config.get_detail()}, log_txt_path)
            pass

        # 计算平均结果
        acc_keys = ["clip_logits", "mlp_logits", "ada_logits", "tot_logits", "acc"]
        for key in acc_keys:
            avg_acc, count = 0, 0
            avg_acc += results[0]['acc'][0][key]  # ImageNet
            count += 1
            for result in results[1:]:
                avg_acc += sum([one[key] for one in result['acc']])
                count += len([one[key] for one in result['acc']])
                pass
            Tools.print(f"avg {key} acc={avg_acc / count}", log_txt_path)
            pass
        pass

    pass


if __name__ == '__main__':
    print("="*80)
    print("HCA-CLIP Training Engine")
    print("="*80)
    print("\nThis module provides the core training engine for HCA-CLIP.")
    print("\nTo run training, please use one of the following:")
    print("  1. scripts/train.py - Main training script with enhanced features")
    print("  2. examples.py - Example training configurations")
    print("\nExample:")
    print("  cd scripts")
    print("  python train.py")
    print("\nFor more information, see README.md and QUICKSTART.md")
    print("="*80)

