import logging

import torch
from torch.utils.data import SequentialSampler

import transformers

logger = logging.getLogger(__name__)


class ProjectionHead(torch.nn.Module):
    """Projection head combining SNV-specific token with multi-pooling sequence context."""
    def __init__(self, input_dim=512, output_dim=2048, snv_pos=None):
        super().__init__()
        self.input_dim = input_dim
        self.snv_pos = snv_pos  # Position of SNV token, if None uses adaptive position
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(input_dim, input_dim, kernel_size=5, padding=0, groups=input_dim),
            torch.nn.Conv1d(input_dim, input_dim, kernel_size=1),
            torch.nn.GELU(),
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(input_dim * 4, output_dim)
        )

    def forward(self, x):
        # x: (B*2, seq_len, D)
        snv_pos = 511

        snv_feat = x[:, snv_pos, :]
        local_feat = self.conv(x[:, snv_pos-2:snv_pos+3, :].permute(0, 2, 1)).squeeze(-1)
        mean_feat = x.mean(dim=1)  
        max_feat = x.max(dim=1)[0]  
        
        # Concatenate: (B, D + D + D + D) = (B, 4*D)
        combined = torch.cat([snv_feat, local_feat, mean_feat, max_feat], dim=-1)
        return self.dense(combined)


class WithProjection(torch.nn.Module):
    """Wrapper model that adds projection head."""
    
    def __init__(self, base_model, input_dim=None, output_dim=2048, model_type=None):
        super().__init__()
        self.base_model = base_model
        self.model_type = model_type
        
        # Auto-detect input dimension if not provided
        if input_dim is None:
            if hasattr(base_model, 'config') and hasattr(base_model.config, 'hidden_size'):
                input_dim = base_model.config.hidden_size
            else:
                input_dim = 768  # Default for most transformers
        
        self.projection_head = ProjectionHead(input_dim, output_dim)
        # Copy config from base model for compatibility
        self.config = base_model.config
    
    def forward(self, input_ids, *args, **kwargs):
        """Forward pass through base model and projection head."""
        # Allow attention_mask to be optional to reduce memory in collated batches
        outputs = self.base_model(input_ids)
        # Extract last_hidden_state and apply projection
        # outputs = outputs.hidden_states[-1]
        outputs = outputs.last_hidden_state
        projected = self.projection_head(outputs)
        return (projected,)


class ContrastiveTrainer(transformers.Trainer):
    """Custom trainer for supervised contrastive learning and mutation regression."""
    def __init__(self, mutation_loss_weight=1.0, clinvar_loss_weight=1.0, pcc_loss_alpha=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mutation_loss_weight = mutation_loss_weight
        self.clinvar_loss_weight = clinvar_loss_weight
        self.pcc_loss_alpha = pcc_loss_alpha
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        self.cos_loss = torch.nn.CosineEmbeddingLoss(-0.8)
        self.triplet_loss = torch.nn.TripletMarginWithDistanceLoss(margin=1.5, distance_function=lambda x, y: 1 - torch.cosine_similarity(x, y))

    def _get_train_sampler(self, train_dataset=None):
        """Use SequentialSampler for monotonously increasing indices."""
        if train_dataset is None:
            train_dataset = self.train_dataset
        return SequentialSampler(train_dataset)

    def _get_eval_sampler(self, eval_dataset):
        """Use SequentialSampler for eval as well."""
        return SequentialSampler(eval_dataset)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute custom contrastive loss for both training and evaluation."""
        labels = inputs.pop("labels")
        batch_type = inputs.pop("batch_type")
        outputs = model(**inputs)

        # Check if we should log this step
        should_log = self.state.global_step % self.args.logging_steps <= 1

        loss = self.contrastive_loss_func(
            outputs, labels, batch_type,
            self.mutation_loss_weight, self.clinvar_loss_weight,
            alpha=self.pcc_loss_alpha,
            should_log=should_log)

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Perform a prediction step using custom contrastive loss for both training and evaluation.
        Overrides parent's prediction_step to use our custom loss function.
        """
        inputs = self._prepare_inputs(inputs)
        labels = inputs.pop("labels")
        batch_type = inputs.pop("batch_type")
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        loss = self.contrastive_loss_func(
            outputs, labels, batch_type,
            self.mutation_loss_weight, self.clinvar_loss_weight,
            alpha=self.pcc_loss_alpha,
            should_log=False
        ) * self.num_gpus
        
        if prediction_loss_only:
            return (loss, None, None)
        
        logits = outputs.get("last_hidden_state") if isinstance(outputs, dict) else (
            outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        )
        
        return (loss, logits, labels)


    def contrastive_loss_func(self, embeddings, labels, batch_type, mutation_loss_weight=1.0, clinvar_loss_weight=1.0, alpha=0.5, should_log=False):
        """Custom loss function for supervised contrastive learning and mutation regression.
        
        batch_type: 0=cd_loss (ref-alt), 1=cdd_loss (benign-pathogenic), 2=pcc_loss (mutation)
        """
        embeddings = embeddings[0].view(labels.shape[0], 3 if batch_type == 1 else 2, -1)

        if batch_type == 0:  # cd_loss: ref-alt comparison
            loss = self.cos_loss(embeddings[:, 0], embeddings[:, 1], labels)
            if should_log:
                logger.info(f"cos_loss: {loss.item():.4f}")
            return clinvar_loss_weight * loss
        
        elif batch_type == 1:  # cdd_loss: benign-pathogenic comparison
            triplet_loss = self.triplet_loss(
                embeddings[:, 0],  # anchor: ref
                embeddings[:, 1],  # positive: benign
                embeddings[:, 2]   # negative: pathogenic
            )
            if should_log:
                logger.info(f"triplet_loss: {triplet_loss.item():.4f}")
            return clinvar_loss_weight * triplet_loss
            
            # https://github.com/GuillaumeErhard/Supervised_contrastive_loss_pytorch/blob/main/loss/spc.py
            # temperature = 0.07
            # projections = embeddings.view(embeddings.shape[0]*2, -1)  # (B*2, D)
            # targets = labels.view(-1)  # (B*2,)
            # dot_product_tempered = torch.mm(projections, projections.T) / temperature
            # # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
            # exp_dot_tempered = (
            #     torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5
            # )

            # mask_similar_class = targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets
            # mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[0])).to(embeddings.device)
            # mask_combined = mask_similar_class * mask_anchor_out
            # cardinality_per_samples = torch.sum(mask_combined, dim=1)

            # log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_anchor_out, dim=1, keepdim=True)))
            # supervised_contrastive_loss_per_sample = torch.sum(log_prob * mask_combined, dim=1) / cardinality_per_samples
            # supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_sample)
            # if should_log:
            #     logger.info(f"cdd_loss: {supervised_contrastive_loss.item():.4f}")
            # return supervised_contrastive_loss


        else:  # bt == 2: pcc_loss for mutation regression
            cos_sim = (1 - torch.cosine_similarity(embeddings[:, 0], embeddings[:, 1], dim=-1)) / 2
            labels = labels / 10
            pcc_loss = (1 - alpha) * torch.nn.functional.mse_loss(cos_sim, labels) + \
                alpha * (1 - torch.nan_to_num(torch.corrcoef(torch.stack([cos_sim, labels]))[0,1])) / 2
            if should_log:
                logger.info(f"pcc_loss: {pcc_loss.item():.4f}")
            return mutation_loss_weight * pcc_loss
