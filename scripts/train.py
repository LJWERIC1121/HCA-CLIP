"""
HCA-CLIP Training Script
Hierarchical Context Aggregation CLIP for Few-Shot Plant Disease Recognition

Features:
- Configurable loss function scales for ablation studies
- Ablation switches for each component
- Detailed logging with hyperparameters
- Support for multiple datasets (Plant Disease, AI Challenger, SPD)
"""

import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ldc_train import Config10Dataset, Runner, LOG_ROOT
from alisuretool.Tools import Tools


class HCAClipConfig:
    """Enhanced configuration for HCA-CLIP with ablation study support"""

    def __init__(self,
                 # Dataset settings
                 dataset_name="plant_disease",
                 shots=16,
                 seed=2024,

                 # Model settings
                 backbone="ViT-B/16",
                 fuse_type=4,  # 4=MDFG, 3=concatenation, 2=addition, 1=single adapter

                 # Training settings
                 train_epoch=50,
                 lr=0.001,
                 batch_size=8,

                 # Loss weights (5 main losses)
                 ce_mlp_weight=1.0,      # CrossEntropy for MLP logits
                 ce_ada_weight=1.0,      # CrossEntropy for Adapted logits
                 ce_tot_weight=1.0,      # CrossEntropy for Total logits
                 l1_mlp_weight=1.0,      # L1 loss for MLP vs CLIP
                 l1_ada_weight=1.0,      # L1 loss for Adapted vs CLIP

                 # Branch classifier losses (for MDFG)
                 ce_1to3_weight=1.0,     # CrossEntropy for branch 1to3
                 ce_2to3_weight=1.0,     # CrossEntropy for branch 2to3
                 l1_1to3_weight=1.0,     # L1 loss for branch 1to3
                 l1_2to3_weight=1.0,     # L1 loss for branch 2to3

                 # Augmentation settings
                 use_rotation=True,
                 rotation_type="rotation8",  # "rotation8", "rotation", "rotation2"

                 # Contrastive learning settings
                 use_contrastive=True,
                 contrastive_weight=0.1,
                 contrastive_temp=0.07,

                 # Ablation switches
                 use_branch_classifiers=True,  # Use intermediate branch classifiers in MDFG
                 ):

        self.dataset_name = dataset_name
        self.shots = shots
        self.seed = seed
        self.backbone = backbone
        self.fuse_type = fuse_type
        self.train_epoch = train_epoch
        self.lr = lr
        self.batch_size = batch_size

        # Loss weights
        self.ce_mlp_weight = ce_mlp_weight
        self.ce_ada_weight = ce_ada_weight
        self.ce_tot_weight = ce_tot_weight
        self.l1_mlp_weight = l1_mlp_weight
        self.l1_ada_weight = l1_ada_weight
        self.ce_1to3_weight = ce_1to3_weight
        self.ce_2to3_weight = ce_2to3_weight
        self.l1_1to3_weight = l1_1to3_weight
        self.l1_2to3_weight = l1_2to3_weight

        # Pack loss_lambda for compatibility
        self.loss_lambda = [ce_mlp_weight, ce_ada_weight, ce_tot_weight, l1_mlp_weight, l1_ada_weight]

        # Augmentation
        self.use_rotation = use_rotation
        self.rotation_type = rotation_type

        # Contrastive learning
        self.use_contrastive = use_contrastive
        self.contrastive_weight = contrastive_weight
        self.contrastive_temp = contrastive_temp

        # Ablation
        self.use_branch_classifiers = use_branch_classifiers

    def to_config10dataset(self):
        """Convert to Config10Dataset for Runner"""
        return Config10Dataset(
            dataset_name=self.dataset_name,
            seed=self.seed,
            shots=self.shots,
            backbone=self.backbone,
            train_epoch=self.train_epoch,
            lr=self.lr,
            batch_size=self.batch_size,
            loss_lambda=self.loss_lambda,
            fuse_type=self.fuse_type,
            use_rotation=self.use_rotation,
            rotation_type=self.rotation_type,
            use_contrastive=self.use_contrastive,
            contrastive_weight=self.contrastive_weight,
            contrastive_temp=self.contrastive_temp
        )

    def get_log_filename(self):
        """Generate descriptive log filename with key hyperparameters"""
        backbone_str = self.backbone.replace('/', '-')

        # Build components for filename
        components = [
            f"{self.dataset_name}",
            f"{self.shots}shot",
            f"{backbone_str}",
            f"bs{self.batch_size}",
            f"lr{self.lr}",
        ]

        # Add loss weights (only if not default)
        if self.loss_lambda != [1.0, 1.0, 1.0, 1.0, 1.0]:
            loss_str = "_".join([f"{w:.1f}" for w in self.loss_lambda])
            components.append(f"loss{loss_str}")

        # Add rotation info
        if self.use_rotation:
            # Generate short names for different augmentation types
            if self.rotation_type == "rotation2":
                components.append("rot2")
            elif self.rotation_type == "rotation":
                components.append("rot4")
            elif self.rotation_type == "rotation8":
                components.append("rot8")
            elif self.rotation_type == "random_augment8":
                components.append("rand8")
            elif self.rotation_type == "random_augment8_strong":
                components.append("rand8s")
            else:
                components.append(f"rot{self.rotation_type}")
        else:
            components.append("norot")

        # Add contrastive info
        if self.use_contrastive:
            components.append(f"con{self.contrastive_weight}")
        else:
            components.append("nocon")

        # Add fusion type
        fuse_names = {1: "single", 2: "add", 3: "cat", 4: "mdfg"}
        components.append(f"fuse{fuse_names.get(self.fuse_type, self.fuse_type)}")

        return "_".join(components) + ".txt"

    def get_summary(self):
        """Get human-readable configuration summary"""
        lines = [
            "="*80,
            "HCA-CLIP Configuration",
            "="*80,
            "",
            "Dataset Settings:",
            f"  Dataset: {self.dataset_name}",
            f"  Shots: {self.shots}",
            f"  Seed: {self.seed}",
            "",
            "Model Settings:",
            f"  Backbone: {self.backbone}",
            f"  Fusion Type: {self.fuse_type} (4=MDFG, 3=cat, 2=add, 1=single)",
            "",
            "Training Settings:",
            f"  Epochs: {self.train_epoch}",
            f"  Learning Rate: {self.lr}",
            f"  Batch Size: {self.batch_size}",
            "",
            "Loss Weights:",
            f"  CE MLP: {self.ce_mlp_weight}",
            f"  CE Adapted: {self.ce_ada_weight}",
            f"  CE Total: {self.ce_tot_weight}",
            f"  L1 MLP: {self.l1_mlp_weight}",
            f"  L1 Adapted: {self.l1_ada_weight}",
        ]

        if self.use_branch_classifiers:
            lines.extend([
                f"  CE Branch 1to3: {self.ce_1to3_weight}",
                f"  CE Branch 2to3: {self.ce_2to3_weight}",
                f"  L1 Branch 1to3: {self.l1_1to3_weight}",
                f"  L1 Branch 2to3: {self.l1_2to3_weight}",
            ])

        lines.extend([
            "",
            "Augmentation:",
            f"  Rotation: {'Enabled' if self.use_rotation else 'Disabled'}",
        ])

        if self.use_rotation:
            lines.append(f"  Rotation Type: {self.rotation_type}")

        lines.extend([
            "",
            "Contrastive Learning:",
            f"  Enabled: {'Yes' if self.use_contrastive else 'No'}",
        ])

        if self.use_contrastive:
            lines.extend([
                f"  Weight: {self.contrastive_weight}",
                f"  Temperature: {self.contrastive_temp}",
            ])

        lines.extend([
            "",
            "Ablation Settings:",
            f"  Use Branch Classifiers: {'Yes' if self.use_branch_classifiers else 'No'}",
            "",
            "="*80,
        ])

        return "\n".join(lines)


class HCAClipExperiments:
    """Experiment runner for HCA-CLIP"""

    def __init__(self):
        self.datasets = {
            "plant_disease": {"name": "plant_disease", "display_name": "Plant Disease", "num_classes": 34},
            "ai_challenger": {"name": "ai_challenger", "display_name": "AI Challenger 2018", "num_classes": 53},
            "spd": {"name": "spd", "display_name": "SPD", "num_classes": 42}
        }

    def run_single_experiment(self, config: HCAClipConfig):
        """Run a single training experiment with given configuration"""

        # Create log filename
        log_filename = config.get_log_filename()
        log_path = Tools.new_dir(os.path.join(LOG_ROOT, log_filename))
        best_results_path = Tools.new_dir(os.path.join(LOG_ROOT, f"best_{log_filename}"))

        print("\n" + config.get_summary())
        print(f"\nLog file: {log_path}")
        print(f"Best results: {best_results_path}")

        # Save config to log
        Tools.print(config.get_summary(), log_path)

        # Convert to Config10Dataset
        train_config = config.to_config10dataset()

        # Run training
        runner = Runner(config=train_config, best_results_log=best_results_path)
        acc_list = runner.train()

        # Save results
        Tools.print(f"\nFinal Results:", log_path)
        Tools.print({"shots": config.shots, "backbone": config.backbone, "acc": acc_list}, log_path)

        print(f"\nResults saved to: {log_path}")
        print(f"Best results saved to: {best_results_path}")

        return acc_list

    def run_all_datasets(self,
                         shots_list=[16],
                         backbone="ViT-B/16",
                         train_epoch=50,
                         lr=0.001,
                         batch_size=8,
                         # Loss weights
                         ce_mlp_weight=1.0,
                         ce_ada_weight=1.0,
                         ce_tot_weight=1.0,
                         l1_mlp_weight=1.0,
                         l1_ada_weight=1.0,
                         ce_1to3_weight=1.0,
                         ce_2to3_weight=1.0,
                         l1_1to3_weight=1.0,
                         l1_2to3_weight=1.0,
                         # Augmentation
                         use_rotation=True,
                         rotation_type="rotation8",
                         # Contrastive learning
                         use_contrastive=True,
                         contrastive_weight=0.1,
                         contrastive_temp=0.07,
                         # Ablation switches
                         fuse_type=4,
                         use_branch_classifiers=True,
                         seed=2024):
        """
        Run experiments for all three datasets with specified configurations

        Args:
            shots_list: List of shot numbers to train (e.g., [1, 2, 4, 8, 16])
            backbone: Model backbone ("ViT-B/16" or "RN50")
            train_epoch: Number of training epochs
            lr: Learning rate
            batch_size: Batch size

            Loss weights:
                ce_mlp_weight: CrossEntropy for MLP logits
                ce_ada_weight: CrossEntropy for Adapted logits
                ce_tot_weight: CrossEntropy for Total logits
                l1_mlp_weight: L1 loss for MLP vs CLIP
                l1_ada_weight: L1 loss for Adapted vs CLIP
                ce_1to3_weight: CrossEntropy for branch 1to3
                ce_2to3_weight: CrossEntropy for branch 2to3
                l1_1to3_weight: L1 loss for branch 1to3
                l1_2to3_weight: L1 loss for branch 2to3

            Augmentation:
                use_rotation: Enable rotation augmentation
                rotation_type: "rotation8", "rotation", or "rotation2"

            Contrastive learning:
                use_contrastive: Enable supervised contrastive loss
                contrastive_weight: Weight for contrastive loss
                contrastive_temp: Temperature parameter

            Ablation switches:
                fuse_type: Feature fusion method (1-4, where 4=MDFG)
                use_branch_classifiers: Enable intermediate classifiers
                seed: Random seed
        """

        all_results = {}

        for dataset_key, dataset_info in self.datasets.items():
            dataset_name = dataset_info["name"]
            display_name = dataset_info["display_name"]

            print(f"\n{'#'*80}")
            print(f"# Starting experiments for: {display_name}")
            print(f"{'#'*80}")

            dataset_results = []

            for shots in shots_list:
                print(f"\n{'='*80}")
                print(f"Dataset: {display_name}, Shots: {shots}")
                print(f"{'='*80}")

                # Create config for this shot
                config = HCAClipConfig(
                    dataset_name=dataset_name,
                    shots=shots,
                    backbone=backbone,
                    train_epoch=train_epoch,
                    lr=lr,
                    batch_size=batch_size,
                    ce_mlp_weight=ce_mlp_weight,
                    ce_ada_weight=ce_ada_weight,
                    ce_tot_weight=ce_tot_weight,
                    l1_mlp_weight=l1_mlp_weight,
                    l1_ada_weight=l1_ada_weight,
                    ce_1to3_weight=ce_1to3_weight,
                    ce_2to3_weight=ce_2to3_weight,
                    l1_1to3_weight=l1_1to3_weight,
                    l1_2to3_weight=l1_2to3_weight,
                    use_rotation=use_rotation,
                    rotation_type=rotation_type,
                    use_contrastive=use_contrastive,
                    contrastive_weight=contrastive_weight,
                    contrastive_temp=contrastive_temp,
                    fuse_type=fuse_type,
                    use_branch_classifiers=use_branch_classifiers,
                    seed=seed
                )

                # Run experiment (no individual log files, only console output)
                train_config = config.to_config10dataset()
                runner = Runner(config=train_config, best_results_log=None)  # No individual best results
                acc_list = runner.train()

                # Store results
                result = {
                    "dataset": display_name,
                    "shots": shots,
                    "acc": acc_list[0] if len(acc_list) > 0 else {}
                }
                dataset_results.append(result)

                # Print all 5 accuracies to console (for shell script to capture)
                if len(acc_list) > 0:
                    best_result = acc_list[0]
                    final_acc = best_result.get('acc', 0.0)
                    clip_acc = best_result.get('clip_logits', 0.0)
                    mlp_acc = best_result.get('mlp_logits', 0.0)
                    ada_acc = best_result.get('ada_logits', 0.0)
                    tot_acc = best_result.get('tot_logits', 0.0)
                    print(f"[RESULT] {dataset_name} {shots}-shot | Final: {final_acc:.2f}% | CLIP: {clip_acc:.2f}% | MLP: {mlp_acc:.2f}% | Adapted: {ada_acc:.2f}% | Total: {tot_acc:.2f}%")

            all_results[dataset_name] = dataset_results

            # Print summary table to console
            print(f"\n{'='*80}")
            print(f"FINAL SUMMARY - {display_name}")
            print(f"{'='*80}")
            print(f"{'Shots':<8} {'Final':<10} {'CLIP':<10} {'MLP':<10} {'Adapted':<10} {'Total':<10}")
            print("-"*60)
            for result in dataset_results:
                acc_data = result['acc']
                print(f"{result['shots']:<8} {acc_data.get('acc', 0.0):<10.2f} {acc_data.get('clip_logits', 0.0):<10.2f} {acc_data.get('mlp_logits', 0.0):<10.2f} {acc_data.get('ada_logits', 0.0):<10.2f} {acc_data.get('tot_logits', 0.0):<10.2f}")
            print("="*80 + "\n")

            print(f"\n{display_name} experiments completed!")

        return all_results

    def run_ablation_study(self, dataset_name="plant_disease", shots=16, backbone="ViT-B/16"):
        """Run ablation study by disabling components one by one"""

        print("\n" + "="*80)
        print("ABLATION STUDY")
        print("="*80)

        ablation_configs = [
            # Full model
            {
                "name": "Full Model",
                "use_rotation": True,
                "use_contrastive": True,
                "fuse_type": 4,
                "use_branch_classifiers": True,
            },
            # Without rotation
            {
                "name": "No Rotation",
                "use_rotation": False,
                "use_contrastive": True,
                "fuse_type": 4,
                "use_branch_classifiers": True,
            },
            # Without contrastive learning
            {
                "name": "No Contrastive",
                "use_rotation": True,
                "use_contrastive": False,
                "fuse_type": 4,
                "use_branch_classifiers": True,
            },
            # Without MDFG (use simple addition)
            {
                "name": "No MDFG",
                "use_rotation": True,
                "use_contrastive": True,
                "fuse_type": 2,  # Simple addition
                "use_branch_classifiers": False,
            },
            # Without branch classifiers
            {
                "name": "No Branch Classifiers",
                "use_rotation": True,
                "use_contrastive": True,
                "fuse_type": 4,
                "use_branch_classifiers": False,
            },
        ]

        results = []

        for ablation in ablation_configs:
            print(f"\n{'='*80}")
            print(f"Ablation: {ablation['name']}")
            print(f"{'='*80}")

            config = HCAClipConfig(
                dataset_name=dataset_name,
                shots=shots,
                backbone=backbone,
                use_rotation=ablation["use_rotation"],
                use_contrastive=ablation["use_contrastive"],
                fuse_type=ablation["fuse_type"],
                use_branch_classifiers=ablation["use_branch_classifiers"],
            )

            acc_list = self.run_single_experiment(config)

            results.append({
                "name": ablation["name"],
                "config": ablation,
                "acc": acc_list[0]["acc"] if len(acc_list) > 0 else 0.0
            })

        # Print ablation summary
        print("\n" + "="*80)
        print("ABLATION STUDY SUMMARY")
        print("="*80)
        for result in results:
            print(f"{result['name']:30s}: {result['acc']:.2f}%")
        print("="*80)

        return results


if __name__ == '__main__':
    experiment = HCAClipExperiments()

    # ===== Example 1: Run single experiment with custom configuration =====
    # config = HCAClipConfig(
    #     dataset_name="plant_disease",
    #     shots=16,
    #     backbone="ViT-B/16",
    #     train_epoch=50,
    #     lr=0.001,
    #     batch_size=8,
    #     # Loss weights
    #     ce_mlp_weight=1.0,
    #     ce_ada_weight=1.0,
    #     ce_tot_weight=1.0,
    #     l1_mlp_weight=1.0,
    #     l1_ada_weight=1.0,
    #     # Augmentation
    #     use_rotation=True,
    #     rotation_type="rotation8",
    #     # Contrastive
    #     use_contrastive=True,
    #     contrastive_weight=0.1,
    #     contrastive_temp=0.07,
    # )
    # experiment.run_single_experiment(config)

    # ===== Example 2: Run all datasets with all hyperparameters =====
    experiment.run_all_datasets(
        shots_list=[1, 2, 4, 8, 16],
        backbone="ViT-B/16",
        train_epoch=1,
        lr=0.001,
        batch_size=8,

        # Loss weights (all set to 1.0 for balanced training)
        ce_mlp_weight=1.0,
        ce_ada_weight=1.0,
        ce_tot_weight=1.0,
        l1_mlp_weight=1.0,
        l1_ada_weight=1.0,
        ce_1to3_weight=1.0,
        ce_2to3_weight=1.0,
        l1_1to3_weight=1.0,
        l1_2to3_weight=1.0,

        # Augmentation settings
        use_rotation=True,
        rotation_type="random_augment8_strong",

        # Contrastive learning settings
        use_contrastive=True,
        contrastive_weight=1,
        contrastive_temp=0.07,

        # Ablation switches
        fuse_type=4,  # MDFG fusion
        use_branch_classifiers=True,
        seed=2024,
    )

    # ===== Example 3: Run ablation study =====
    # experiment.run_ablation_study(
    #     dataset_name="plant_disease",
    #     shots=16,
    #     backbone="ViT-B/16"
    # )
