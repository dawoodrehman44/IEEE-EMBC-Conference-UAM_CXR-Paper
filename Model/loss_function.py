class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    Focuses training on hard-to-classify examples
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='none'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits, targets):
        """
        Args:
            logits: [B, num_classes]
            targets: [B, num_classes]
        """
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        
        # Focal weight: (1 - pt)^gamma
        focal_weight = (1 - pt) ** self.gamma
        
        # Final loss
        loss = self.alpha * focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class EnhancedLoss(nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()
        # Replace BCE with Focal Loss
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0, reduction='none')
        self.class_weights = class_weights  # [num_classes] tensor
        
    def classification_loss_with_uncertainty(self, logits, log_var, labels):
        """
        Uncertainty-aware classification with Focal Loss + Class Weighting
        """
        # Focal loss per sample per class [B, num_classes]
        focal_loss = self.focal_loss(logits, labels)
        
        # Apply class weights if provided
        if self.class_weights is not None:
            focal_loss = focal_loss * self.class_weights.unsqueeze(0)
        
        # Clamp log_var to prevent numerical issues
        log_var = torch.clamp(log_var, -5, 5)
        
        # Precision (inverse variance)
        precision = torch.exp(-log_var)
        
        # Uncertainty-weighted loss
        loss = 0.5 * (precision * focal_loss + log_var)
        
        return loss.mean()
    
    def contrastive_loss(self, vision_feat, text_feat, temperature=0.07):
        batch_size = vision_feat.size(0)
        
        # Use projected features (same dimension: 128)
        # Normalize features
        vision_feat = F.normalize(vision_feat, dim=-1)
        text_feat = F.normalize(text_feat, dim=-1)
        
        logits = torch.matmul(vision_feat, text_feat.T) / temperature
        labels = torch.arange(batch_size).to(vision_feat.device)
        
        loss_v2t = F.cross_entropy(logits, labels)
        loss_t2v = F.cross_entropy(logits.T, labels)
        
        return (loss_v2t + loss_t2v) / 2
    
    def conformal_coverage_loss(self, conformal_scores, labels, logits, quantile=None):
        """
        FIXED: Learnable conformal loss with explicit coverage target
        
        Goals:
        1. High error → high score
        2. Low error → low score  
        3. Maintain ~85% coverage (scores distributed around quantile)
        """
        probs = torch.sigmoid(logits)
        batch_size = len(labels)
        
        # 1. Classification error (target for score)
        classification_error = ((probs - labels) ** 2).mean(dim=1)  # [B]
        
        # 2. Ranking loss: score should match error
        ranking_loss = F.mse_loss(conformal_scores, classification_error.detach())
        
        # 3. Coverage loss: encourage score distribution to enable 85% coverage
        if quantile is not None:
            # We want ~15% of scores > quantile (for miscoverage)
            # and ~85% of scores <= quantile (for coverage)
            
            # Count how many scores exceed quantile
            exceed_rate = (conformal_scores > quantile).float().mean()
            target_exceed_rate = 0.15  # Want 15% miscoverage
            
            # Penalize deviation from target
            coverage_loss = (exceed_rate - target_exceed_rate) ** 2
        else:
            coverage_loss = 0.0
        
        # 4. Score regularization: prevent collapse to 0
        score_reg = torch.abs(conformal_scores.mean() - 0.5)  # Keep scores centered around 0.5
        
        total_loss = ranking_loss + 0.5 * coverage_loss + 0.1 * score_reg
        
        return total_loss
    
    def uncertainty_regularization(self, log_var):
        """
        Regularize uncertainty to avoid:
        - Too confident (log_var → -∞)
        - Too uncertain (log_var → +∞)
        """
        # Encourage moderate uncertainty
        return torch.abs(log_var).mean()
    
    def coverage_penalty_loss(self, conformal_scores, labels, logits, target_coverage=0.85):
        """
        Direct coverage loss - penalize when coverage is too low
        """
        probs = torch.sigmoid(logits)
        batch_size = len(labels)
        
        # For each sample, check if it would be covered
        covered = []
        for i in range(batch_size):
            # Simulate prediction set based on score
            score = conformal_scores[i]
            
            # Lower score → smaller set
            # We want: low score for correct predictions (covered)
            #         high score for wrong predictions (needs bigger set)
            
            error = ((probs[i] - labels[i]) ** 2).mean()
            
            # If error is low and score is low → covered (good)
            # If error is high and score is high → covered (good, will get big set)
            # If error is high and score is low → NOT covered (bad!)
            
            is_covered = (score > error).float()  # Heuristic: score should exceed error
            covered.append(is_covered)
        
        coverage = torch.stack(covered).mean()
        
        # Penalize if coverage < target
        coverage_loss = F.relu(target_coverage - coverage)
        
        return coverage_loss
    
    def forward(self, outputs, labels, quantile=None):
        """
        Compute total loss
        
        Args:
            outputs: Dict from model forward pass
            labels: Ground truth
            quantile: Current conformal quantile (if available)
        
        Returns:
            total_loss, loss_dict
        """
        # 1. Classification loss (with uncertainty)
        cls_loss = self.classification_loss_with_uncertainty(
            outputs['logits'],
            outputs['log_var'],
            labels
        )
        
        # 2. Contrastive loss (use same-dimension features)
        contrast_loss = self.contrastive_loss(
            outputs['fused_feat'],  # Use fused features (both 512)
            outputs['fused_feat']   # Or project vision/text separately
        )
        
        # 3. Conformal coverage loss
        conformal_loss = self.conformal_coverage_loss(
            outputs['conformal_scores'],
            labels,
            outputs['logits'],
            quantile
        )
        
        # 4. Uncertainty regularization
        uncertainty_reg = self.uncertainty_regularization(outputs['log_var'])
        
        # Add to loss computation
        coverage_penalty = self.coverage_penalty_loss(
            outputs['conformal_scores'],
            labels,
            outputs['logits'],
            target_coverage=0.85
        )

        total_loss = (
            config.lambda_cls * cls_loss +
            config.lambda_contrast * contrast_loss +
            config.lambda_conformal * (conformal_loss + coverage_penalty) +  # CHANGED
            config.lambda_uncertainty * uncertainty_reg
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'classification': cls_loss.item(),
            'contrastive': contrast_loss.item(),
            'conformal': conformal_loss.item(),
            'uncertainty_reg': uncertainty_reg.item()
        }
        
        return total_loss, loss_dict