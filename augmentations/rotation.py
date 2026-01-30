"""
Multi-View Data Augmentation for Few-Shot Learning
Includes rotation-based and random augmentation strategies
Adapted from VOCP project
"""
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import random


def rotation2():
    """
    Apply rotation augmentation: 0° and 180° rotations
    Returns:
        transform_func: function to apply rotation
        multiplier: number of augmented views per image (2)
    """
    def _transform(images):
        """
        Args:
            images: [B, C, H, W] tensor
        Returns:
            rotated_images: [B*2, C, H, W] tensor
        """
        size = images.shape[1:]  # (C, H, W)
        # Stack original (k=0) and 180° rotated (k=2) images
        return torch.stack([torch.rot90(images, k, (2, 3)) for k in [0, 2]], 1).view(-1, *size)

    return _transform, 2


def rotation():
    """
    Apply rotation augmentation: 0°, 90°, 180°, 270° rotations
    Returns:
        transform_func: function to apply rotation
        multiplier: number of augmented views per image (4)
    """
    def _transform(images):
        """
        Args:
            images: [B, C, H, W] tensor
        Returns:
            rotated_images: [B*4, C, H, W] tensor
        """
        size = images.shape[1:]  # (C, H, W)
        # Stack all 4 rotations
        return torch.stack([torch.rot90(images, k, (2, 3)) for k in range(4)], 1).view(-1, *size)

    return _transform, 4


def rotation8():
    """
    Apply rotation augmentation: 8 rotations at 45° intervals
    0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°
    Returns:
        transform_func: function to apply rotation
        multiplier: number of augmented views per image (8)
    """
    def _transform(images):
        """
        Args:
            images: [B, C, H, W] tensor
        Returns:
            rotated_images: [B*8, C, H, W] tensor
        """
        B, C, H, W = images.shape
        rotated_list = []

        # Angles: 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°
        angles = [0, 45, 90, 135, 180, 225, 270, 315]

        for angle in angles:
            # Use TF.rotate for arbitrary angles
            # torchvision's rotate works on batched tensors
            rotated = TF.rotate(images, angle=angle, interpolation=TF.InterpolationMode.BILINEAR)
            rotated_list.append(rotated)

        # Stack: [B, 8, C, H, W] -> [B*8, C, H, W]
        return torch.stack(rotated_list, dim=1).view(B * 8, C, H, W)

    return _transform, 8


def random_rotation2(max_angle=180):
    """
    Apply random rotation augmentation: 2 random angles per image
    Args:
        max_angle: maximum absolute rotation angle in degrees
    Returns:
        transform_func: function to apply random rotation
        multiplier: number of augmented views per image (2)
    """
    def _transform(images):
        """
        Args:
            images: [B, C, H, W] tensor
        Returns:
            rotated_images: [B*2, C, H, W] tensor
        """
        B, C, H, W = images.shape
        rotated_list = []

        for _ in range(2):
            angle = random.uniform(-max_angle, max_angle)
            rotated = TF.rotate(images, angle=angle, interpolation=TF.InterpolationMode.BILINEAR)
            rotated_list.append(rotated)

        return torch.stack(rotated_list, dim=1).view(B * 2, C, H, W)

    return _transform, 2


def random_rotation4(max_angle=180):
    """
    Apply random rotation augmentation: 4 random angles per image
    Args:
        max_angle: maximum absolute rotation angle in degrees
    Returns:
        transform_func: function to apply random rotation
        multiplier: number of augmented views per image (4)
    """
    def _transform(images):
        """
        Args:
            images: [B, C, H, W] tensor
        Returns:
            rotated_images: [B*4, C, H, W] tensor
        """
        B, C, H, W = images.shape
        rotated_list = []

        for _ in range(4):
            angle = random.uniform(-max_angle, max_angle)
            rotated = TF.rotate(images, angle=angle, interpolation=TF.InterpolationMode.BILINEAR)
            rotated_list.append(rotated)

        return torch.stack(rotated_list, dim=1).view(B * 4, C, H, W)

    return _transform, 4


def random_rotation8(max_angle=180):
    """
    Apply random rotation augmentation: 8 random angles per image
    Args:
        max_angle: maximum absolute rotation angle in degrees
    Returns:
        transform_func: function to apply random rotation
        multiplier: number of augmented views per image (8)
    """
    def _transform(images):
        """
        Args:
            images: [B, C, H, W] tensor
        Returns:
            rotated_images: [B*8, C, H, W] tensor
        """
        B, C, H, W = images.shape
        rotated_list = []

        for _ in range(8):
            angle = random.uniform(-max_angle, max_angle)
            rotated = TF.rotate(images, angle=angle, interpolation=TF.InterpolationMode.BILINEAR)
            rotated_list.append(rotated)

        return torch.stack(rotated_list, dim=1).view(B * 8, C, H, W)

    return _transform, 8


def get_rotation_labels(labels, multiplier):
    """
    Expand labels to match augmented images
    Args:
        labels: [B] original labels
        multiplier: number of augmentations per image
    Returns:
        expanded_labels: [B*multiplier] labels
    """
    return labels.repeat_interleave(multiplier)


def random_augment8():
    """
    Apply random augmentation: 8 different random augmented views
    Each view uses different random transformations including:
    - Random rotation
    - Random horizontal flip
    - Random crop and resize
    - Random color jitter
    - Random grayscale

    Returns:
        transform_func: function to apply random augmentation
        multiplier: number of augmented views per image (8)
    """
    def _transform(images):
        """
        Args:
            images: [B, C, H, W] tensor (already normalized)
        Returns:
            augmented_images: [B*8, C, H, W] tensor
        """
        B, C, H, W = images.shape
        augmented_list = []

        # Define augmentation parameters
        rotation_angles = [-30, -15, 0, 15, 30, 45, -45, 0]  # 8 different angles
        flip_probs = [0.0, 0.5, 1.0, 0.5, 0.0, 0.5, 0.5, 0.0]  # Different flip probabilities

        for i in range(8):
            # Start with original images
            aug_images = images.clone()

            # Random rotation (different angle for each view)
            if rotation_angles[i] != 0:
                aug_images = TF.rotate(
                    aug_images,
                    angle=rotation_angles[i],
                    interpolation=TF.InterpolationMode.BILINEAR
                )

            # Random horizontal flip (different probability for each view)
            if random.random() < flip_probs[i]:
                aug_images = TF.hflip(aug_images)

            # Random crop and resize (different scale for each view)
            if i % 2 == 0:  # Apply to half of the views
                # Random resized crop parameters
                scale_min = 0.7 + (i * 0.05)  # Different scales: 0.7, 0.8, 0.9, 1.0
                scale_max = min(1.0, scale_min + 0.2)

                # Apply random crop for each image in batch
                aug_images_list = []
                for j in range(B):
                    img = aug_images[j]
                    # Random crop
                    i_param, j_param, h_param, w_param = T.RandomResizedCrop.get_params(
                        img, scale=(scale_min, scale_max), ratio=(0.9, 1.1)
                    )
                    img = TF.crop(img, i_param, j_param, h_param, w_param)
                    img = TF.resize(img, [H, W], interpolation=TF.InterpolationMode.BILINEAR)
                    aug_images_list.append(img)
                aug_images = torch.stack(aug_images_list)

            # Random color jitter (different strength for each view)
            if i in [1, 3, 5, 7]:  # Apply to half of the views
                brightness = 0.1 + (i * 0.05)
                contrast = 0.1 + (i * 0.05)
                saturation = 0.1 + (i * 0.05)

                # Apply color jitter for each image in batch
                aug_images_list = []
                for j in range(B):
                    img = aug_images[j]
                    # Random brightness
                    if random.random() < 0.8:
                        factor = random.uniform(1 - brightness, 1 + brightness)
                        img = TF.adjust_brightness(img, factor)
                    # Random contrast
                    if random.random() < 0.8:
                        factor = random.uniform(1 - contrast, 1 + contrast)
                        img = TF.adjust_contrast(img, factor)
                    # Random saturation
                    if random.random() < 0.8:
                        factor = random.uniform(1 - saturation, 1 + saturation)
                        img = TF.adjust_saturation(img, factor)
                    aug_images_list.append(img)
                aug_images = torch.stack(aug_images_list)

            # Random grayscale (only for views 2 and 6)
            if i in [2, 6]:
                if random.random() < 0.2:  # 20% chance
                    aug_images_list = []
                    for j in range(B):
                        img = aug_images[j]
                        # Convert to grayscale and back to RGB
                        gray = TF.rgb_to_grayscale(img, num_output_channels=3)
                        aug_images_list.append(gray)
                    aug_images = torch.stack(aug_images_list)

            augmented_list.append(aug_images)

        # Stack: [B, 8, C, H, W] -> [B*8, C, H, W]
        return torch.stack(augmented_list, dim=1).view(B * 8, C, H, W)

    return _transform, 8


def random_augment8_strong():
    """
    Apply strong random augmentation: 8 different strongly augmented views
    Stronger version of random_augment8 with more aggressive transformations

    Returns:
        transform_func: function to apply random augmentation
        multiplier: number of augmented views per image (8)
    """
    def _transform(images):
        """
        Args:
            images: [B, C, H, W] tensor (already normalized)
        Returns:
            augmented_images: [B*8, C, H, W] tensor
        """
        B, C, H, W = images.shape
        augmented_list = []

        for i in range(8):
            # Start with original images
            aug_images = images.clone()

            # Stronger random rotation
            angle = random.uniform(-45, 45)
            aug_images = TF.rotate(
                aug_images,
                angle=angle,
                interpolation=TF.InterpolationMode.BILINEAR
            )

            # Random horizontal flip (50% chance)
            if random.random() < 0.5:
                aug_images = TF.hflip(aug_images)

            # Stronger random crop and resize
            aug_images_list = []
            for j in range(B):
                img = aug_images[j]
                # Aggressive crop
                i_param, j_param, h_param, w_param = T.RandomResizedCrop.get_params(
                    img, scale=(0.5, 1.0), ratio=(0.75, 1.33)
                )
                img = TF.crop(img, i_param, j_param, h_param, w_param)
                img = TF.resize(img, [H, W], interpolation=TF.InterpolationMode.BILINEAR)

                # Stronger color jitter
                if random.random() < 0.8:
                    img = TF.adjust_brightness(img, random.uniform(0.6, 1.4))
                if random.random() < 0.8:
                    img = TF.adjust_contrast(img, random.uniform(0.6, 1.4))
                if random.random() < 0.8:
                    img = TF.adjust_saturation(img, random.uniform(0.6, 1.4))
                if random.random() < 0.8:
                    img = TF.adjust_hue(img, random.uniform(-0.1, 0.1))

                # Random grayscale
                if random.random() < 0.2:
                    img = TF.rgb_to_grayscale(img, num_output_channels=3)

                aug_images_list.append(img)

            aug_images = torch.stack(aug_images_list)
            augmented_list.append(aug_images)

        # Stack: [B, 8, C, H, W] -> [B*8, C, H, W]
        return torch.stack(augmented_list, dim=1).view(B * 8, C, H, W)

    return _transform, 8

