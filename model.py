import logging
import torch
from torch.utils.data import SequentialSampler
import transformers

logger = logging.getLogger(__name__)


class ProjectionHead(torch.nn.Module):
    """Enhanced projection head with multi-layer max-pooling, MHA, and SNV-focused features."""
    
    def __init__(self, input_dim=512, output_dim=2048, snv_pos=511, num_heads=8):
        super().__init__()
        self.input_dim = input_dim
        self.snv_pos = snv_pos
        
        self.mha = torch.nn.MultiheadAttention(input_dim, num_heads, batch_first=True)
        self.layer_norm_mha = torch.nn.LayerNorm(input_dim)
        
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(input_dim, input_dim, kernel_size=5, padding=0, groups=input_dim),
            torch.nn.Conv1d(input_dim, input_dim, kernel_size=1),
            torch.nn.GELU(),
        )
        
        self.dense = torch.nn.Linear(input_dim * 4, output_dim)
        
    def forward(self, hidden_states_list):
        stacked = torch.stack(hidden_states_list, dim=1)
        max_pooled = stacked.max(dim=1)[0]
        mean_pooled = stacked.mean(dim=1)

        snv_feat = mean_pooled[:, self.snv_pos, :] 
        local_feat = self.conv(mean_pooled[:, self.snv_pos-2:self.snv_pos+3, :].permute(0, 2, 1)).squeeze(-1)
        
        aggregated = self.layer_norm_mha(self.mha(max_pooled, max_pooled, max_pooled)[0])

        combined = torch.cat([aggregated.mean(dim=1), aggregated.max(dim=1)[0], snv_feat, local_feat], dim=-1)
        return self.dense(combined)

    
class WrapperModel(torch.nn.Module):
    
    def __init__(self, base_model, input_dim=None, output_dim=2048, model_type=None, 
                 num_heads=8, num_layers=4, snv_pos=511):
        super().__init__()
        self.base_model = base_model
        self.model_type = model_type
        self.num_layers = num_layers
        
        if input_dim is None:
            input_dim = getattr(base_model.config, 'hidden_size', 768)
        
        self.projection_head = ProjectionHead(input_dim, output_dim, snv_pos=snv_pos, num_heads=num_heads)
        self.config = base_model.config
        
        if hasattr(self.config, 'output_hidden_states'):
            self.config.output_hidden_states = True
    
    def forward(self, input_ids, *args, **kwargs):
        kwargs['output_hidden_states'] = True
        outputs = self.base_model(input_ids, **kwargs)
        
        hidden_states_list = (list(outputs.hidden_states[-self.num_layers:]) 
                             if hasattr(outputs, 'hidden_states') and outputs.hidden_states 
                             else [outputs.last_hidden_state])
        
        projected = (self.projection_head(hidden_states_list))
        
        return (projected,)


class ContrastiveTrainer(transformers.Trainer):
    """Custom trainer for supervised contrastive learning and mutation regression."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        self.margin = getattr(self.args, "cos_loss_margin", 0.8)
        self.mse_loss = torch.nn.MSELoss()
        self.cos_loss = torch.nn.CosineEmbeddingLoss(margin=-self.margin)

    def _get_train_sampler(self, train_dataset=None):
        return SequentialSampler(train_dataset or self.train_dataset)

    def _get_eval_sampler(self, eval_dataset):
        return SequentialSampler(eval_dataset) if eval_dataset else None
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        eval_dataset = eval_dataset or self.eval_dataset
        if eval_dataset is None:
            logger.info("No eval dataset provided, skipping evaluation")
            return {}
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        batch_type = inputs.pop("batch_type")
        outputs = model(**inputs)

        loss = self.contrastive_loss_func(
            outputs, labels, batch_type,
            should_log=self.state.global_step % self.args.logging_steps <= 1)

        return (loss, outputs) if return_outputs else loss

    def contrastive_loss_func(self, embeddings, labels, batch_type, should_log=False, is_train=True):
        embeddings = embeddings[0].view(labels.shape[0], 2, -1)

        if batch_type == 0:
            loss = torch.cosine_similarity(embeddings[:, 0], embeddings[:, 1], dim=-1)
            loss = (loss[labels == -1].sum() + (loss[labels == 1] - self.margin).abs().sum()) / labels.shape[0]
            # loss = self.cos_loss(embeddings[:, 0], -embeddings[:, 1], -labels)
            if should_log:
                logger.info(f"cos_loss: {loss.item():.4f}")
            return loss
        else:
            cd = (1 - torch.cosine_similarity(embeddings[:, 0], embeddings[:, 1], dim=-1))/2
            loss = self.mse_loss(cd, labels) # - torch.corrcoef(torch.stack([cd, labels]))[0, 1]
            if should_log:
                logger.info(f"pearson_loss: {loss.item():.4f}")
            return loss 
