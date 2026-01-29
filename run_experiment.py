"""
Command-line training script for HCA-CLIP hyperparameter search
Supports flexible parameter configuration for large-scale experiments
"""

import os
import sys
import argparse

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from scripts.train import HCAClipExperiments


def parse_args():
    parser = argparse.ArgumentParser(description='HCA-CLIP Hyperparameter Search')

    # Dataset settings
    parser.add_argument('--datasets', type=str, default='plant_disease,ai_challenger,spd',
                        help='Comma-separated list of datasets')
    parser.add_argument('--shots', type=str, default='1,2,4,8,16',
                        help='Comma-separated list of shot numbers')
    parser.add_argument('--backbone', type=str, default='ViT-B/16',
                        help='Model backbone')
    parser.add_argument('--train_epoch', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')

    # Loss weights
    parser.add_argument('--ce_mlp_weight', type=float, default=1.0)
    parser.add_argument('--ce_ada_weight', type=float, default=1.0)
    parser.add_argument('--ce_tot_weight', type=float, default=1.0)
    parser.add_argument('--l1_mlp_weight', type=float, default=1.0)
    parser.add_argument('--l1_ada_weight', type=float, default=1.0)
    parser.add_argument('--ce_1to3_weight', type=float, default=1.0)
    parser.add_argument('--ce_2to3_weight', type=float, default=1.0)
    parser.add_argument('--l1_1to3_weight', type=float, default=1.0)
    parser.add_argument('--l1_2to3_weight', type=float, default=1.0)

    # Augmentation
    parser.add_argument('--use_rotation', type=int, default=1,
                        help='Use rotation augmentation (0 or 1)')
    parser.add_argument('--rotation_type', type=str, default='rotation8',
                        choices=['rotation2', 'rotation', 'rotation8',
                                'random_augment8', 'random_augment8_strong'])

    # Contrastive learning
    parser.add_argument('--use_contrastive', type=int, default=1,
                        help='Use contrastive learning (0 or 1)')
    parser.add_argument('--contrastive_weight', type=float, default=0.1)
    parser.add_argument('--contrastive_temp', type=float, default=0.07)

    # Ablation
    parser.add_argument('--fuse_type', type=int, default=4)
    parser.add_argument('--use_branch_classifiers', type=int, default=1,
                        help='Use branch classifiers (0 or 1)')
    parser.add_argument('--seed', type=int, default=2024)

    # Experiment identification
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Optional experiment name for logging')

    return parser.parse_args()


def main():
    args = parse_args()

    # Parse datasets and shots
    datasets = [d.strip() for d in args.datasets.split(',')]
    shots_list = [int(s.strip()) for s in args.shots.split(',')]

    # Convert 0/1 to bool
    use_rotation = bool(args.use_rotation)
    use_contrastive = bool(args.use_contrastive)
    use_branch_classifiers = bool(args.use_branch_classifiers)

    # Print experiment configuration
    print("="*80)
    print("HCA-CLIP Hyperparameter Search Experiment")
    print("="*80)
    if args.exp_name:
        print(f"Experiment Name: {args.exp_name}")
    print(f"Datasets: {datasets}")
    print(f"Shots: {shots_list}")
    print(f"Backbone: {args.backbone}")
    print(f"Training Epochs: {args.train_epoch}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print("\nLoss Weights:")
    print(f"  CE MLP: {args.ce_mlp_weight}")
    print(f"  CE Adapted: {args.ce_ada_weight}")
    print(f"  CE Total: {args.ce_tot_weight}")
    print(f"  L1 MLP: {args.l1_mlp_weight}")
    print(f"  L1 Adapted: {args.l1_ada_weight}")
    print(f"  CE 1to3: {args.ce_1to3_weight}")
    print(f"  CE 2to3: {args.ce_2to3_weight}")
    print(f"  L1 1to3: {args.l1_1to3_weight}")
    print(f"  L1 2to3: {args.l1_2to3_weight}")
    print("\nAugmentation:")
    print(f"  Use Rotation: {use_rotation}")
    print(f"  Rotation Type: {args.rotation_type}")
    print("\nContrastive Learning:")
    print(f"  Use Contrastive: {use_contrastive}")
    print(f"  Weight: {args.contrastive_weight}")
    print(f"  Temperature: {args.contrastive_temp}")
    print("="*80)
    print()

    # Create experiment runner
    experiment = HCAClipExperiments()

    # Filter datasets
    experiment.datasets = {k: v for k, v in experiment.datasets.items() if k in datasets}

    # Run experiments
    results = experiment.run_all_datasets(
        shots_list=shots_list,
        backbone=args.backbone,
        train_epoch=args.train_epoch,
        lr=args.lr,
        batch_size=args.batch_size,

        # Loss weights
        ce_mlp_weight=args.ce_mlp_weight,
        ce_ada_weight=args.ce_ada_weight,
        ce_tot_weight=args.ce_tot_weight,
        l1_mlp_weight=args.l1_mlp_weight,
        l1_ada_weight=args.l1_ada_weight,
        ce_1to3_weight=args.ce_1to3_weight,
        ce_2to3_weight=args.ce_2to3_weight,
        l1_1to3_weight=args.l1_1to3_weight,
        l1_2to3_weight=args.l1_2to3_weight,

        # Augmentation
        use_rotation=use_rotation,
        rotation_type=args.rotation_type,

        # Contrastive learning
        use_contrastive=use_contrastive,
        contrastive_weight=args.contrastive_weight,
        contrastive_temp=args.contrastive_temp,

        # Ablation
        fuse_type=args.fuse_type,
        use_branch_classifiers=use_branch_classifiers,
        seed=args.seed,
    )

    print("\n" + "="*80)
    print("Experiment Completed Successfully!")
    print("="*80)
    print(f"\nResults saved to: result/log/")
    print(f"Check summary files: summary_*_all_shots_*.txt")
    print("="*80)

    return results


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
