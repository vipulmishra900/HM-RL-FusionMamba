import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

from datasets.custom_dataset import PairedImageDataset
from FusionMamba import FusionMamba
from state_encoder import MultiModalEncoder
from PPO_agent import ActorCritic, PPOAlgorithm
from adaptive_controller import AdaptiveController
from reward_function import MultiModalRewardShaper


class Trainer:
    """
    Supervised trainer for FusionMamba with iteration-level checkpoint/resume.

    Training can safely stop after max_iterations NEW iterations.
    Example:
        Run 1 -> steps 1-100
        Run 2 -> steps 101-200
        Run 3 -> steps 201-300
    """

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.training.device)

        # ---------------------------------------------------------
        # 1. Initialize Networks
        # ---------------------------------------------------------
        self.encoder = MultiModalEncoder(
            visual_shape=config.encoder.visual_input_shape,
            proprio_dim=config.encoder.proprioceptive_dim,
            embed_dim=config.encoder.embed_dim
        ).to(self.device)

        self.fusion_mamba = FusionMamba(
            in_channels=1,
            base_channels=32,
            out_channels=1
        ).to(self.device)

        self.actor_critic = ActorCritic(
            state_dim=config.mamba.d_model,
            action_dim=config.agent.action_dim,
            hidden_dim=config.agent.hidden_dim,
            action_space_type=config.agent.action_space_type
        ).to(self.device)

        # PPO kept for compatibility
        self.ppo = PPOAlgorithm(self.actor_critic, config)

        # ---------------------------------------------------------
        # 2. Controller & Reward
        # ---------------------------------------------------------
        self.controller = AdaptiveController(config)
        self.reward_shaper = MultiModalRewardShaper()

        # ---------------------------------------------------------
        # 3. Dataset
        # ---------------------------------------------------------
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        if hasattr(config.training, "dataset_dir") and config.training.dataset_dir:
            llvip_dir = config.training.dataset_dir
        elif os.environ.get("DATASET_DIR"):
            llvip_dir = os.environ.get("DATASET_DIR")
        else:
            llvip_dir = os.path.join(base_dir, "datasets", "LLVIP")

        resize_shape = config.encoder.visual_input_shape[1:]

        batch_size = getattr(
            config.training,
            "batch_size",
            config.agent.mini_batch_size
        )

        # ---------------------------------------------------------
        # 4. Directories
        # ---------------------------------------------------------
        os.makedirs(config.training.checkpoint_dir, exist_ok=True)
        os.makedirs(config.training.log_dir, exist_ok=True)

        # ---------------------------------------------------------
        # 5. Training Dataset
        # ---------------------------------------------------------
        self.dataset = PairedImageDataset(
            root_dir=llvip_dir,
            resize_shape=resize_shape,
            convert_to_grayscale=True,
            is_train=True,
            split="train"
        )

        if len(self.dataset) == 0:
            print(
                f"WARNING: No image pairs found in '{llvip_dir}'."
            )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=(len(self.dataset) > 0),
            num_workers=0
        )

        # ---------------------------------------------------------
        # 6. Validation Dataset
        # ---------------------------------------------------------
        self.val_dataset = PairedImageDataset(
            root_dir=llvip_dir,
            resize_shape=resize_shape,
            convert_to_grayscale=True,
            is_train=False,
            split="test"
        )

        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )

        # ---------------------------------------------------------
        # 7. Optimizer
        # ---------------------------------------------------------
        lr = getattr(config.training, "lr", 1e-4)

        self.optimizer = optim.Adam(
            self.fusion_mamba.parameters(),
            lr=lr
        )

        # ---------------------------------------------------------
        # 8. Training State
        # ---------------------------------------------------------
        self.start_epoch = 1
        self.start_iteration = 1

        # Total number of iterations completed across ALL runs
        self.global_step = 0

        self.best_val_loss = float("inf")

        # ---------------------------------------------------------
        # 9. Load Checkpoint (Auto-resume latest checkpoint if exists)
        # ---------------------------------------------------------
        checkpoint_path = getattr(
            config.training,
            "checkpoint_path",
            None
        )

        auto_resume = getattr(config.training, "auto_resume", True)
        latest_checkpoint_file = os.path.join(
            config.training.checkpoint_dir,
            "latest_checkpoint.pth"
        )

        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        elif auto_resume and os.path.exists(latest_checkpoint_file):
            print(
                f"\nAuto-resume: Found latest checkpoint at '{latest_checkpoint_file}'. Resuming..."
            )
            self.load_checkpoint(latest_checkpoint_file)

    # =============================================================
    # SSIM LOSS
    # =============================================================

    def ssim_loss(self, img1, img2, window_size=11):

        import torch.nn.functional as F

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu1 = F.avg_pool2d(
            img1,
            window_size,
            stride=1,
            padding=window_size // 2
        )

        mu2 = F.avg_pool2d(
            img2,
            window_size,
            stride=1,
            padding=window_size // 2
        )

        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = torch.clamp(
            F.avg_pool2d(
                img1 * img1,
                window_size,
                stride=1,
                padding=window_size // 2
            ) - mu1_sq,
            min=0.0
        )

        sigma2_sq = torch.clamp(
            F.avg_pool2d(
                img2 * img2,
                window_size,
                stride=1,
                padding=window_size // 2
            ) - mu2_sq,
            min=0.0
        )

        sigma12 = (
            F.avg_pool2d(
                img1 * img2,
                window_size,
                stride=1,
                padding=window_size // 2
            )
            - mu1_mu2
        )

        num = (
            (2.0 * mu1_mu2 + C1)
            * (2.0 * sigma12 + C2)
        )

        den = (
            (mu1_sq + mu2_sq + C1)
            * (sigma1_sq + sigma2_sq + C2)
        )

        ssim_map = num / (den + 1e-10)

        return 1.0 - ssim_map.mean()

    # =============================================================
    # COMPATIBILITY
    # =============================================================

    def collect_rollouts(self, num_steps):

        states_list = []
        actions_list = []
        log_probs_list = []
        returns_list = []
        advantages_list = []

        return (
            states_list,
            actions_list,
            log_probs_list,
            returns_list,
            advantages_list
        )

    # =============================================================
    # SAVE CHECKPOINT
    # =============================================================

    def save_checkpoint(self, epoch, iteration, path):

        checkpoint = {
            "epoch": epoch,
            "iteration": iteration,
            "global_step": self.global_step,
            "global_iteration": self.global_step,

            "model_state_dict":
                self.fusion_mamba.state_dict(),

            "optimizer_state_dict":
                self.optimizer.state_dict(),

            "best_val_loss":
                self.best_val_loss
        }

        torch.save(checkpoint, path)

        print(
            f"Checkpoint saved successfully: {path}"
        )

    # =============================================================
    # LOAD CHECKPOINT
    # =============================================================

    def load_checkpoint(self, path):

        if not os.path.exists(path):

            print(
                f"WARNING: Checkpoint not found at {path}. "
                "Starting from scratch."
            )

            return

        checkpoint = torch.load(
            path,
            map_location=self.device
        )

        # Restore model
        self.fusion_mamba.load_state_dict(
            checkpoint["model_state_dict"]
        )

        # Restore optimizer
        self.optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        self.best_val_loss = checkpoint.get(
            "best_val_loss",
            float("inf")
        )

        # ---------------------------------------------------------
        # New checkpoint format
        # ---------------------------------------------------------

        has_global_iter = (
            "global_iteration" in checkpoint
            or "global_step" in checkpoint
        )

        if (
            has_global_iter
            and "iteration" in checkpoint
        ):

            self.global_step = checkpoint.get(
                "global_iteration",
                checkpoint.get("global_step", 0)
            )

            saved_epoch = checkpoint["epoch"]
            saved_iteration = checkpoint["iteration"]

            # If entire epoch was completed
            if len(self.dataloader) > 0 and saved_iteration >= len(self.dataloader):

                self.start_epoch = saved_epoch + 1
                self.start_iteration = 1

            else:

                self.start_epoch = saved_epoch
                self.start_iteration = saved_iteration + 1

            print(
                "\n========== CHECKPOINT RESUMED =========="
            )

            print(
                f"Previous Epoch    : {saved_epoch}"
            )

            print(
                f"Previous Iteration: {saved_iteration}"
            )

            print(
                f"Global Iteration  : {self.global_step}"
            )

            print(
                f"Resume Epoch      : {self.start_epoch}"
            )

            print(
                f"Resume Iteration  : {self.start_iteration}"
            )

            print(
                "========================================\n"
            )

        # ---------------------------------------------------------
        # Old checkpoint compatibility
        # ---------------------------------------------------------

        else:

            self.start_epoch = (
                checkpoint["epoch"] + 1
            )

            self.start_iteration = 1
            self.global_step = 0

            print(
                "Old checkpoint detected. "
                f"Resuming from epoch {self.start_epoch}."
            )

    # =============================================================
    # VALIDATION
    # =============================================================

    def validate(self):

        if len(self.val_dataloader) == 0:

            print(
                "Validation DataLoader is empty. "
                "Skipping validation."
            )

            return float("nan")

        print(
            "Starting validation loop..."
        )

        self.fusion_mamba.eval()

        total_val_loss = 0.0

        with torch.no_grad():

            for iteration, batch in enumerate(
                self.val_dataloader,
                1
            ):

                ir = batch["ir"].to(self.device)
                vis = batch["vis"].to(self.device)

                fused = self.fusion_mamba(
                    ir,
                    vis
                )

                # L1
                loss_l1 = (
                    torch.mean(
                        torch.abs(fused - ir)
                    )
                    +
                    torch.mean(
                        torch.abs(fused - vis)
                    )
                ) / 2.0

                # SSIM
                loss_ssim = (
                    self.ssim_loss(
                        fused,
                        ir
                    )
                    +
                    self.ssim_loss(
                        fused,
                        vis
                    )
                ) / 2.0

                loss = (
                    0.8 * loss_l1
                    +
                    0.2 * loss_ssim
                )

                total_val_loss += loss.item()

        avg_val_loss = (
            total_val_loss
            /
            len(self.val_dataloader)
        )

        print(
            f"Validation Loss: {avg_val_loss:.6f}"
        )

        self.fusion_mamba.train()

        return avg_val_loss

    # =============================================================
    # TRAIN
    # =============================================================

    def train(self):

        if len(self.dataloader) == 0:

            raise RuntimeError(
                f"Training DataLoader is empty. "
                f"No image pairs found in "
                f"'{self.dataset.root_dir}'."
            )

        print(
            "Starting supervised FusionMamba "
            "training pipeline..."
        )

        self.fusion_mamba.train()

        total_epochs = getattr(
            self.config.training,
            "epochs",
            5
        )

        if self.start_epoch > total_epochs:
            print(
                f"\nTraining already completed all {total_epochs} epochs "
                f"(current start epoch is {self.start_epoch}). "
                f"Increase --epochs to train further, or use --no-resume to start fresh."
            )
            return

        # Number of NEW iterations to perform THIS RUN
        max_iterations = getattr(
            self.config.training,
            "max_iterations",
            None
        )

        # ---------------------------------------------------------
        # IMPORTANT:
        # max_iterations means NEW iterations this run.
        # global_step remains cumulative.
        # ---------------------------------------------------------

        iterations_this_run = 0

        for epoch in range(
            self.start_epoch,
            total_epochs + 1
        ):

            total_train_loss = 0.0
            batches_processed = 0

            start_iteration = (
                self.start_iteration
                if epoch == self.start_epoch
                else 1
            )

            for iteration, batch in enumerate(
                self.dataloader,
                1
            ):

                # Skip already processed batches
                if iteration < start_iteration:
                    continue

                ir = batch["ir"].to(
                    self.device
                )

                vis = batch["vis"].to(
                    self.device
                )

                # -------------------------------------------------
                # Forward
                # -------------------------------------------------

                fused = self.fusion_mamba(
                    ir,
                    vis
                )

                # -------------------------------------------------
                # L1 Loss
                # -------------------------------------------------

                loss_l1 = (
                    torch.mean(
                        torch.abs(fused - ir)
                    )
                    +
                    torch.mean(
                        torch.abs(fused - vis)
                    )
                ) / 2.0

                # -------------------------------------------------
                # SSIM Loss
                # -------------------------------------------------

                loss_ssim = (
                    self.ssim_loss(
                        fused,
                        ir
                    )
                    +
                    self.ssim_loss(
                        fused,
                        vis
                    )
                ) / 2.0

                # -------------------------------------------------
                # Total Loss
                # -------------------------------------------------

                loss = (
                    0.8 * loss_l1
                    +
                    0.2 * loss_ssim
                )

                # -------------------------------------------------
                # Backpropagation
                # -------------------------------------------------

                self.optimizer.zero_grad()

                loss.backward()

                self.optimizer.step()

                # -------------------------------------------------
                # Update Counters
                # -------------------------------------------------

                total_train_loss += loss.item()

                batches_processed += 1

                self.global_step += 1

                iterations_this_run += 1

                # -------------------------------------------------
                # Log
                # -------------------------------------------------

                print(
                    f"Epoch: {epoch} | "
                    f"Iteration: {iteration}/{len(self.dataloader)} | "
                    f"Global Step: {self.global_step} | "
                    f"Loss: {loss.item():.6f}"
                )

                # -------------------------------------------------
                # STOP AFTER N NEW ITERATIONS
                # -------------------------------------------------

                if (
                    max_iterations is not None
                    and iterations_this_run
                    >= max_iterations
                ):

                    latest_path = os.path.join(
                        self.config.training.checkpoint_dir,
                        "latest_checkpoint.pth"
                    )

                    self.save_checkpoint(
                        epoch,
                        iteration,
                        latest_path
                    )

                    print(
                        "\n========================================"
                    )

                    print(
                        f"Reached {max_iterations} NEW "
                        "iterations for this run."
                    )

                    print(
                        f"Global Iteration: {self.global_step}"
                    )

                    print(
                        f"Saved at Epoch {epoch}, "
                        f"Iteration {iteration}"
                    )

                    print(
                        "Next run will resume from the "
                        "next iteration."
                    )

                    print(
                        "========================================\n"
                    )

                    return

            # -----------------------------------------------------
            # Epoch Completed
            # -----------------------------------------------------

            self.start_iteration = 1

            if batches_processed == 0:
                continue

            avg_train_loss = (
                total_train_loss
                /
                batches_processed
            )

            print(
                f"Epoch {epoch} Training Completed. "
                f"Average Train Loss: "
                f"{avg_train_loss:.6f}"
            )

            # -----------------------------------------------------
            # Validation
            # -----------------------------------------------------

            val_loss = self.validate()

            print(
                f"Epoch {epoch} Summary | "
                f"Train Loss: {avg_train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f}"
            )

            # -----------------------------------------------------
            # Best Model
            # -----------------------------------------------------

            if (
                not np.isnan(val_loss)
                and val_loss < self.best_val_loss
            ):

                self.best_val_loss = val_loss

                best_path = os.path.join(
                    self.config.training.checkpoint_dir,
                    "best_model.pth"
                )

                self.save_checkpoint(
                    epoch,
                    len(self.dataloader),
                    best_path
                )

                print(
                    f"New best model found at "
                    f"epoch {epoch}!"
                )

            # -----------------------------------------------------
            # Latest Checkpoint
            # -----------------------------------------------------

            latest_path = os.path.join(
                self.config.training.checkpoint_dir,
                "latest_checkpoint.pth"
            )

            self.save_checkpoint(
                epoch,
                len(self.dataloader),
                latest_path
            )

        print(
            "Supervised training loop "
            "completed successfully!"
        )
