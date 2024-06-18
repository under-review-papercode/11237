import gc
import random
from typing import List, Tuple, Union
import torch
from torch import Tensor
from torch import optim
import wandb
from .models import Vector_VQVAE, VQ_SVG_Stage2
import pytorch_lightning as pl
from utils import log_images, log_all_images, get_side_by_side_reconstruction, add_points_to_image, get_merged_image_for_logging
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.multimodal.clip_score import CLIPScore
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import StepLR
import pandas as pd
from torchmetrics.functional.multimodal import clip_score
from tokenizer import VQTokenizer
import torch_optimizer as optim_

class SVG_VQVAE_Stage2_Experiment(pl.LightningModule):
    def __init__(self,
                 model: VQ_SVG_Stage2,
                 tokenizer: VQTokenizer,
                 num_batches_train: int,
                 num_batches_val: int,
                 lr: float = 0.0003,
                 weight_decay: float = 0.0,
                 scheduler_gamma: float = 0.99,
                 train_log_interval: float = 0.05,
                 val_log_interval: float = 0.1,
                 metric_log_interval: float = 0.1,
                 manual_seed: int = 42,
                 wandb: bool = False,
                 post_process: bool = True,
                 **kwargs) -> None:
        super(SVG_VQVAE_Stage2_Experiment, self).__init__()

        self.model = model
        self.tokenizer = tokenizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_gamma = scheduler_gamma
        assert train_log_interval < 1 and train_log_interval >= 0, f"train log interval should be a fraction of the total number of batches in [0, 1), got {train_log_interval}"
        assert metric_log_interval < 1 and metric_log_interval >= 0, f"metric log interval should be a fraction of the total number of batches in [0, 1), got {metric_log_interval}"
        self.train_log_interval = max(1, int(train_log_interval * num_batches_train))
        self.val_log_interval = max(1, int(val_log_interval * num_batches_val))
        self.train_metric_log_interval = max(1, int(metric_log_interval * num_batches_train))
        self.val_metric_log_interval = max(1, int(metric_log_interval * num_batches_val))
        self.manual_seed = manual_seed
        self.curr_device = None
        self.wandb = wandb
        self.post_process = post_process

    def forward(self, text_tokens: Tensor, text_attention_mask: Tensor, vq_tokens: Tensor, logging=False,
                **kwargs) -> list:
        out, logging_dict = self.model.forward(text_tokens, text_attention_mask, vq_tokens, logging=logging, **kwargs)
        return out, logging_dict

    def _generate_rasterized_sample(self, text_tokens: Tensor, text_attention_mask: Tensor, vq_tokens: Tensor,
                                    temperature:float = 0.0, sampling_method: str = None, sampling_kwargs:dict = {},
                                    post_process: bool = True, draw_context_red: bool = True) -> Tensor:
        """
        Args:
            - text_tokens (Tensor): (1, t)
            - text_attention_mask (Tensor): (1, t)
            - vq_tokens (Tensor): (1, input_context_len_you_want)
        """
        num_input_context_tokens = vq_tokens.shape[-1] // 2
        with torch.no_grad():
            generation, reason = self.model.generate(text_tokens, text_attention_mask, vq_tokens,
                                                     temperature=temperature, sampling_method=sampling_method,sampling_kwargs=sampling_kwargs)
            if generation.ndim > 1:
                generation = generation[0]
        if draw_context_red:
            return self.tokenizer._tokens_to_image_tensor(generation, post_process=post_process,
                                                          num_strokes_to_paint=num_input_context_tokens)
        else:
            return self.tokenizer._tokens_to_image_tensor(generation, post_process=post_process)

    def _get_clip_score_for_batch(self, text_tokens: Tensor, text_attention_mask: Tensor, vq_tokens: Tensor,
                                  post_process: bool = True, temperatures:List=None) -> Tuple[Tensor, List, List]:
        """
        gets clip scores for 0-context generations of a batch of text tokens
        """
        with torch.no_grad():
            bs = text_tokens.shape[0]
            texts = [self.tokenizer.decode_text(text_tokens[i]) for i in range(bs)]
            # filter out empty texts
            relevant_idxs = [i for i in range(bs) if len(texts[i]) > 0]
            generations = [
                self._generate_rasterized_sample(
                    text_tokens[i:i + 1, :],
                    text_attention_mask[i:i + 1, :],
                    vq_tokens[i:i + 1, :1],
                    post_process=post_process,
                    temperature=temperatures[i] if temperatures is not None else 0.0
                ).to(self.curr_device) for i in relevant_idxs
            ]
            texts = [texts[i] for i in relevant_idxs]
            metric = clip_score(generations, texts, "openai/clip-vit-base-patch16")
        return metric, generations, texts

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        text_tokens, text_attention_mask, vq_tokens, vq_targets, pad_token = batch
        self.curr_device = text_tokens.device

        if batch_idx % self.train_log_interval == 0 and self.wandb:
            out, logging_dict = self.forward(text_tokens, text_attention_mask, vq_tokens, logging=True)
            text_condition = self.tokenizer.decode_text(text_tokens[0])
            rasterized_gt = self.tokenizer._tokens_to_image_tensor(vq_targets[:1], post_process=self.post_process)

            # every third batch use temp = 0
            if batch_idx % (self.train_log_interval * 3) == 0:
                temperature = 0.0
            else:
                temperature = random.uniform(0.2, 1.5)

            context_0_generation = self._generate_rasterized_sample(text_tokens[:1, :], text_attention_mask[:1, :],
                                                                    vq_tokens[:1, :1], post_process=self.post_process,
                                                                    temperature=temperature)
            context_5_generation = self._generate_rasterized_sample(text_tokens[:1, :], text_attention_mask[:1, :],
                                                                    vq_tokens[:1, :6], post_process=self.post_process,
                                                                    temperature=temperature)
            context_10_generation = self._generate_rasterized_sample(text_tokens[:1, :], text_attention_mask[:1, :],
                                                                     vq_tokens[:1, :11], post_process=self.post_process,
                                                                     temperature=temperature)
            images = [rasterized_gt, context_0_generation, context_5_generation, context_10_generation]
            self.trainer.logger.log_image(
                key="train/rasterized_samples",
                caption=[f"GT: {text_condition}, temp: {round(temperature, ndigits=2)}, VQ context is marked red."] * len(images),
                images=images,
            )
        else:
            out, logging_dict = self.forward(text_tokens, text_attention_mask, vq_tokens, logging=False)

        if batch_idx % self.train_metric_log_interval == 0 and self.wandb:
            with torch.no_grad():
                num_samples = 8
                temperatures = [random.uniform(0.0, 1.5) for _ in range(num_samples)]
                clip_score_metric, generations, texts = self._get_clip_score_for_batch(
                    text_tokens[:num_samples],
                    text_attention_mask[:num_samples],
                    vq_tokens[:num_samples],
                    post_process=self.post_process,
                    temperatures=temperatures
                )
                self.log("train/clip_score", clip_score_metric, rank_zero_only=True, logger=True, on_step=True)
                self.trainer.logger.log_image(
                    key="train/generated_samples",
                    caption=[text+f", temp: {round(temperatures[i], ndigits=2)}" for i, text in enumerate(texts)],
                    images=generations,
                )

        pred_logits = out  # (b, vq_token_len)
        pred_logits = pred_logits.reshape(-1, pred_logits.shape[-1])

        targets = vq_targets.view(-1)

        # mask out pad token for loss calculation
        mask = targets != pad_token[0]
        pred_logits = pred_logits[mask]
        targets = targets[mask]

        # This is logging a table of tokens to the wandb dashboard
        if batch_idx % self.train_log_interval == 0 and self.wandb:
            target_unique_values, target_counts = torch.unique(targets.detach().cpu(), return_counts=True)
            pred_unique_values, pred_counts = torch.unique(pred_logits.detach().cpu().argmax(dim=1), return_counts=True)
            df = pd.DataFrame(zip(target_unique_values.tolist(), target_counts.tolist()),
                              columns=["token_idx", "target_count"])
            df_pred = pd.DataFrame(zip(pred_unique_values.tolist(), pred_counts.tolist()),
                                   columns=["token_idx", "pred_count"])
            df = pd.merge(df, df_pred, on='token_idx', how='outer').fillna(0)
            df["target_count"] = df["target_count"].astype(int)
            df["pred_count"] = df["pred_count"].astype(int)
            sorted_df = df.sort_values(by='target_count', ascending=False).reset_index(drop=True)
            self.trainer.logger.log_table("train/target_pred_token_counts", dataframe=sorted_df)

        loss_dict = self.model.loss_function(
            targets=targets,
            pred_logits=pred_logits,
        )

        self.log_dict(loss_dict, logger=True, rank_zero_only=True)
        self.log("train_loss", loss_dict["loss"].detach().item(), rank_zero_only=True)
        return loss_dict["loss"]

    def on_train_epoch_end(self):
        # gc.collect()
        # torch.cuda.empty_cache()
        return {}

    def validation_step(self, batch, batch_idx, optimizer_idx=0):

        text_tokens, text_attention_mask, vq_tokens, vq_targets, pad_token = batch
        self.curr_device = text_tokens.device

        with torch.no_grad():
            if batch_idx % self.train_log_interval == 0 and self.wandb:
                out, logging_dict = self.forward(text_tokens, text_attention_mask, vq_tokens, logging=True)
                text_condition = self.tokenizer.decode_text(text_tokens[0])

                # every third batch use temp = 0
                if batch_idx % (self.train_log_interval * 3) == 0:
                    temperature = 0.0
                else:
                    temperature = random.uniform(0.2, 1.5)

                rasterized_gt = self.tokenizer._tokens_to_image_tensor(vq_targets[:1], post_process=self.post_process)
                context_0_generation = self._generate_rasterized_sample(text_tokens[:1, :], text_attention_mask[:1, :],
                                                                        vq_tokens[:1, :1],
                                                                        post_process=self.post_process,
                                                                        temperature=temperature)
                context_5_generation = self._generate_rasterized_sample(text_tokens[:1, :], text_attention_mask[:1, :],
                                                                        vq_tokens[:1, :6],
                                                                        post_process=self.post_process,
                                                                        temperature=temperature)
                context_10_generation = self._generate_rasterized_sample(text_tokens[:1, :], text_attention_mask[:1, :],
                                                                         vq_tokens[:1, :11],
                                                                         post_process=self.post_process,
                                                                         temperature=temperature)
                
                images = [rasterized_gt, context_0_generation, context_5_generation, context_10_generation]
                self.trainer.logger.log_image(
                    key="val/rasterized_samples",
                    caption=[f"GT: {text_condition}, temp: {round(temperature, ndigits=2)}, VQ context is marked red."] * len(images),
                    images=images,
                )
            else:
                out, logging_dict = self.forward(text_tokens, text_attention_mask, vq_tokens, logging=False)

            if batch_idx % self.val_metric_log_interval == 0 and self.wandb:
                num_samples = 8
                temperatures = [random.uniform(0.0, 1.5) for _ in range(num_samples)]
                clip_score_metric, generations, texts = self._get_clip_score_for_batch(
                    text_tokens[:num_samples],
                    text_attention_mask[:num_samples],
                    vq_tokens[:num_samples],
                    post_process=self.post_process,
                    temperatures=temperatures
                )
                self.log("val/clip_score", clip_score_metric, rank_zero_only=True, logger=True)
                self.trainer.logger.log_image(
                    key="val/generated_samples",
                    caption=[text+f", temp: {round(temperatures[i], ndigits=2)}" for i, text in enumerate(texts)],
                    images=generations,
                )

        pred_logits = out  # (b, vq_token_len)
        pred_logits = pred_logits.reshape(-1, pred_logits.shape[-1])

        targets = vq_targets.view(-1)
        # mask out pad token for loss calculation
        mask = targets != pad_token[0]
        pred_logits = pred_logits[mask]
        targets = targets[mask]

        if batch_idx % self.train_log_interval == 0 and self.wandb:
            target_unique_values, target_counts = torch.unique(targets.detach().cpu(), return_counts=True)
            pred_unique_values, pred_counts = torch.unique(pred_logits.detach().cpu().argmax(dim=1), return_counts=True)
            df = pd.DataFrame(zip(target_unique_values.tolist(), target_counts.tolist()),
                              columns=["token_idx", "target_count"])
            df_pred = pd.DataFrame(zip(pred_unique_values.tolist(), pred_counts.tolist()),
                                   columns=["token_idx", "pred_count"])
            df = pd.merge(df, df_pred, on='token_idx', how='outer').fillna(0)
            df["target_count"] = df["target_count"].astype(int)
            sorted_df = df.sort_values(by='target_count', ascending=False).reset_index(drop=True)
            self.trainer.logger.log_table("val/target_pred_token_counts", dataframe=sorted_df)

        loss_dict = self.model.loss_function(
            targets=targets,
            pred_logits=pred_logits,
        )

        self.log("val_loss", loss_dict["loss"], sync_dist=True)
        return loss_dict["loss"]

    def on_validation_end(self) -> None:
        # if self.wandb:
        #     self.sample_images()
        # gc.collect()
        # torch.cuda.empty_cache()
        return {}

    def configure_optimizers(self):

        optims = []
        scheds = []

        param_group_1 = {'params': self.model.parameters(), 'lr': self.lr}
        param_groups = [param_group_1]

        if not self.weight_decay:
            optimizer = optim.AdamW(
                param_groups,
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        else:
            # learning rates should be explicitly specified in the param_groups
            optimizer = optim.Adam(param_groups)
        optims.append(optimizer)

        try:
            if self.scheduler_gamma is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma=self.scheduler_gamma)
                scheds.append(scheduler)

                return optims, scheds
        except:
            pass
        return optims


class VectorVQVAE_Experiment_Stage1(pl.LightningModule):
    """
    Vector quantized pre-training of an autoencoder for SVG primitives.
    """

    def __init__(self,
                 model: Vector_VQVAE,
                 vector_decoder_model: str = "mlp",  # or mlp
                 lr: float = 0.0003,
                 weight_decay: float = 0.0,
                 scheduler_gamma: float = 0.99,
                 train_log_interval: float = 0.05,
                 val_log_interval:float = 0.1,
                 manual_seed: int = 42,
                 min_lr: float = 1.e-6,
                 step_lr_epoch_step_size: int = 30,
                 scheduler_type: str = "cosine",
                 wandb: bool = True,
                 datamodule = None,
                 max_epochs:int=300,
                 **kwargs) -> None:
        super(VectorVQVAE_Experiment_Stage1, self).__init__()

        assert train_log_interval < 1 and train_log_interval >= 0, f"train log interval should be a fraction of the total number of batches in [0, 1), got {train_log_interval}"
        # assert metric_log_interval < 1 and metric_log_interval >= 0, f"metric log interval should be a fraction of the total number of batches in [0, 1), got {metric_log_interval}"
        # self.train_log_interval = max(1, int(train_log_interval * num_batches_train))
        self.num_batches_train = len(datamodule.train_dataloader())
        self.num_batches_val = len(datamodule.val_dataloader())

        self.model = model
        self.vector_decoder_model = vector_decoder_model
        self.lr = lr
        self.total_steps = max_epochs * self.num_batches_train
        self.min_lr = min_lr
        self.weight_decay = weight_decay
        self.scheduler_gamma = scheduler_gamma
        self.train_log_interval =  max(1, int(train_log_interval * self.num_batches_train))
        self.val_log_interval = max(1, int(val_log_interval * self.num_batches_val))
        self.manual_seed = manual_seed
        self.curr_device = None
        self.wandb = wandb
        self.datamodule = datamodule
        self.scheduler_type = scheduler_type
        self.step_size = step_lr_epoch_step_size

    def forward(self, input_images: Tensor, logging=False,**kwargs) -> list:
        out, logging_dict = self.model.forward(input_images, logging=logging, **kwargs)
        return out, logging_dict
    
    def training_step(self, batch, batch_idx, optimizer_idx=0):
        all_center_shapes, labels, centers, descriptions = batch
        self.curr_device = all_center_shapes.device
        bs = all_center_shapes.shape[0]
        channels = all_center_shapes.shape[1]
        if batch_idx % self.train_log_interval == 0 and self.wandb:
            out, logging_dict = self.forward(all_center_shapes, logging=True)
        else:
            out, logging_dict = self.forward(all_center_shapes, logging=False)  # out is [reconstructions, input, all_points, vq_loss]
        reconstructions=out[0]
        inputs = all_center_shapes
        all_points = out[2]
        vq_loss=out[3]

        loss_dict = self.model.loss_function(
            reconstructions=reconstructions[:,:channels,:,:],
            gt_images=inputs,
            vq_loss=vq_loss,
            points=all_points,
        )
    
        # always log the first batch and variable amount of timesteps up to 10
        if batch_idx % self.train_log_interval == 0 and self.wandb:
            with torch.no_grad():
                logging_dict = {f"train/{key}": value for key, value in logging_dict.items()}
                wandb.log(logging_dict)
                random_idx = random.randint(0, len(self.datamodule.train_dataset))
                side_by_side_recons = get_side_by_side_reconstruction(self.model, self.datamodule.train_dataset, idx = random_idx, device = self.curr_device)
                wandb.log({"train/side_by_side_recons":wandb.Image(side_by_side_recons, caption="side by side reconstructions of training sample")})
                if reconstructions.shape[0] > 25:
                    log_amount = 25
                else:
                    log_amount = reconstructions.shape[0]

                log_reconstructions = add_points_to_image(all_points, reconstructions[:,:3,:,:], image_scale=reconstructions.shape[-1])

                # Log input against prediction
                log_images(
                    log_reconstructions[:log_amount],
                    inputs[:log_amount],
                    log_key="train/reconstruction",
                    captions="input (left) vs. reconstruction (right)"
                )

        self.log_dict(loss_dict, logger=True)
        return loss_dict["loss"]


    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        with torch.no_grad():
            all_center_shapes, label, centers, descriptions = batch
            self.curr_device = all_center_shapes.device
            bs = all_center_shapes.shape[0]
            channels = all_center_shapes.shape[1]

            out, logging_dict = self.forward(all_center_shapes)
            reconstructions=out[0]
            inputs = all_center_shapes
            all_points = out[2]
            vq_loss=out[3]
            assert vq_loss.dim() <= 1, f"vq_loss should be a 1D tensor, but got {vq_loss.dim()}"

            loss_dict = self.model.loss_function(
                reconstructions=reconstructions[:,:channels,:,:],
                gt_images=inputs,
                vq_loss=vq_loss,
                points=all_points,
            )
            # log_reconstructions = add_points_to_image(all_points, reconstructions[:,:3,:,:], image_scale=reconstructions.shape[-1])
            if batch_idx % self.val_log_interval == 0 and self.wandb:
                logging_dict = {f"val/{key}": value for key, value in logging_dict.items()}
                wandb.log(logging_dict)
                random_idx = random.randint(0, len(self.datamodule.val_dataset))
                side_by_side_recons = get_side_by_side_reconstruction(self.model, self.datamodule.val_dataset, idx = random_idx, device = self.curr_device)
                wandb.log({"val/side_by_side_recons":wandb.Image(side_by_side_recons, caption="side by side reconstructions of validation sample")})
                if reconstructions.shape[0] > 25:
                    log_amount = 25
                else:
                    log_amount = reconstructions.shape[0]

                log_reconstructions = add_points_to_image(all_points[:log_amount], reconstructions[:log_amount,:3,:,:], image_scale=reconstructions.shape[-1])
                # Log input against prediction
                log_images(
                    log_reconstructions[:log_amount],
                    inputs[:log_amount],
                    log_key="val/reconstruction",
                    captions="input (left) vs. reconstruction (right)"
                )

        self.log("val_loss", loss_dict["loss"])
        return loss_dict["loss"]

    
    def configure_optimizers(self):

        optims = []
        scheds = []

        param_group_1 = {'params': self.model.parameters(), 'lr': self.lr}
        param_groups = [param_group_1]

        if not self.weight_decay:
            optimizer = optim.AdamW(
                param_groups,
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        else:
            # learning rates should be explicitly specified in the param_groups
            optimizer = optim.Adam(param_groups)
        optims.append(optimizer)

        if self.scheduler_type == "cosine":
            scheds.append(CosineAnnealingLR(optimizer, T_max=self.total_steps, eta_min=self.min_lr))
            return optims, scheds
        elif self.scheduler_type == "step":
            scheds.append(StepLR(optimizer, step_size=self.step_size, gamma=self.scheduler_gamma))
            return optims, scheds
        elif self.scheduler_type == "exponential":
            try:
                if self.scheduler_gamma is not None:
                    scheduler = optim.lr_scheduler.ExponentialLR(optims[0], gamma = self.scheduler_gamma)
                    scheds.append(scheduler)
                    return optims, scheds
            except:
                return optims
        elif self.scheduler_type == "none":
            return optims
        else:
            raise Exception(f"Unknown scheduler for this training: {self.scheduler_type}")

    # def on_train_batch_end(self, output, batch, batch_index):
    #     # Perform evaluation after every eval_steps steps
    #     if batch_index % self.eval_steps == 0:
    #         self.trainer.fit_loop.epoch_loop.val_loop.run()
