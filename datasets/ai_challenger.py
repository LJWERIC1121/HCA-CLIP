import os
import json
from collections import defaultdict

from .utils import Datum, DatasetBase, read_json, write_json, build_data_loader


# Simple template for baseline comparison
simple_template = "a photo of {}"


class AiChallenger(DatasetBase):

    dataset_dir = 'Ai_Challenger_2018'

    def __init__(self, root, num_shots, use_rotation_augmentation=False):
        self.dataset_dir = root
        self.num_shots = num_shots
        self.use_rotation_augmentation = use_rotation_augmentation

        # Load metadata
        metadata_path = os.path.join(self.dataset_dir, 'metadata.json')
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        # Load classnames
        classnames_path = os.path.join(self.dataset_dir, 'classnames.txt')
        with open(classnames_path, 'r') as f:
            lines = f.readlines()
            self._class_list = [line.strip() for line in lines if line.strip()]

        # Use simple template for text prompts
        self.template = simple_template

        # Read train and test data (no val, consistent with plant_disease)
        train = self.read_data(f'train_{num_shots}shot', apply_rotation=use_rotation_augmentation)
        test = self.read_data('test', apply_rotation=False)  # No augmentation for test

        super().__init__(train_x=train, val=test, test=test)  # Use test as val for compatibility

    def read_data(self, split_dir, apply_rotation=False):
        """Read data from train_Nshot, val, or test directory"""
        split_path = os.path.join(self.dataset_dir, split_dir)
        items = []

        if not os.path.exists(split_path):
            print(f"Warning: {split_path} does not exist")
            return items

        # Iterate through class directories
        class_dirs = sorted(os.listdir(split_path))
        for class_dir in class_dirs:
            class_path = os.path.join(split_path, class_dir)
            if not os.path.isdir(class_path):
                continue

            # Get label and classname
            classname = class_dir.replace('_', ' ')

            # Find label from classnames list
            try:
                label = self._class_list.index(class_dir)
            except ValueError:
                print(f"Warning: {class_dir} not found in classnames.txt")
                continue

            # Read all images in this class directory
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_path, img_name)

                    if apply_rotation:
                        # Create multiple rotation versions as separate samples
                        # rotation2: 0° and 180°
                        for rotation_id in [0, 2]:  # 0=0°, 2=180°
                            item = Datum(
                                impath=img_path,
                                label=label,
                                classname=classname,
                                rotation=rotation_id  # Add rotation marker
                            )
                            items.append(item)
                    else:
                        # No rotation, just add original image
                        item = Datum(
                            impath=img_path,
                            label=label,
                            classname=classname
                        )
                        items.append(item)

        if apply_rotation:
            print(f"Loaded {len(items)} images from {split_dir} (with rotation augmentation, multiplier=2)")
        else:
            print(f"Loaded {len(items)} images from {split_dir}")
        return items
