class VisionOnlyModel(nn.Module):
    """Ablation 1: Vision encoder only, no text"""
    
    def __init__(self):
        super().__init__()
        self.vision_encoder = VisionEncoder(embed_dim=768)
        self.classifier = UncertaintyAwareClassifier(input_dim=768, num_classes=14)
    
    def forward(self, images, token_ids=None):
        vision_feat = self.vision_encoder(images)
        logits, log_var = self.classifier(vision_feat)
        
        # Dummy features for compatibility
        text_feat = torch.zeros(images.size(0), 512, device=images.device)
        fused_feat = vision_feat[:, :512] if vision_feat.size(1) >= 512 else F.pad(vision_feat, (0, 512-vision_feat.size(1)))
        
        return {
            'logits': logits,
            'log_var': log_var,
            'vision_feat': vision_feat,
            'text_feat': text_feat,
            'fused_feat': fused_feat,
            'conformal_scores': torch.zeros(images.size(0), device=images.device)
        }


class TextOnlyModel(nn.Module):
    """Ablation 2: Text encoder only, no vision"""
    
    def __init__(self, vocab_size):
        super().__init__()
        self.text_encoder = TextEncoder(vocab_size=vocab_size, embed_dim=512, num_layers=3)
        self.classifier = UncertaintyAwareClassifier(input_dim=512, num_classes=14)
    
    def forward(self, images, token_ids):
        text_feat = self.text_encoder(token_ids)
        logits, log_var = self.classifier(text_feat)
        
        # Dummy features
        vision_feat = torch.zeros(images.size(0), 768, device=images.device)
        fused_feat = text_feat
        
        return {
            'logits': logits,
            'log_var': log_var,
            'vision_feat': vision_feat,
            'text_feat': text_feat,
            'fused_feat': fused_feat,
            'conformal_scores': torch.zeros(images.size(0), device=images.device)
        }


class NoFusionModel(nn.Module):
    """Ablation 3: Separate vision and text, no fusion"""
    
    def __init__(self, vocab_size):
        super().__init__()
        self.vision_encoder = VisionEncoder(embed_dim=768)
        self.text_encoder = TextEncoder(vocab_size=vocab_size, embed_dim=512, num_layers=3)
        
        # Separate classifiers
        self.vision_classifier = UncertaintyAwareClassifier(input_dim=768, num_classes=14)
        self.text_classifier = UncertaintyAwareClassifier(input_dim=512, num_classes=14)
    
    def forward(self, images, token_ids):
        vision_feat = self.vision_encoder(images)
        text_feat = self.text_encoder(token_ids)
        
        # Average predictions from both
        v_logits, v_log_var = self.vision_classifier(vision_feat)
        t_logits, t_log_var = self.text_classifier(text_feat)
        
        logits = (v_logits + t_logits) / 2
        log_var = (v_log_var + t_log_var) / 2
        
        fused_feat = torch.cat([vision_feat, text_feat], dim=1)
        
        return {
            'logits': logits,
            'log_var': log_var,
            'vision_feat': vision_feat,
            'text_feat': text_feat,
            'fused_feat': fused_feat,
            'conformal_scores': torch.zeros(images.size(0), device=images.device)
        }


class SimpleFusionModel(nn.Module):
    """Ablation 4: Concatenation fusion instead of attention"""
    
    def __init__(self, vocab_size):
        super().__init__()
        self.vision_encoder = VisionEncoder(embed_dim=768)
        self.text_encoder = TextEncoder(vocab_size=vocab_size, embed_dim=512, num_layers=3)
        
        # Simple concatenation fusion
        self.fusion = nn.Sequential(
            nn.Linear(768 + 512, 512),
            nn.ReLU(),
            nn.LayerNorm(512)
        )
        
        self.classifier = UncertaintyAwareClassifier(input_dim=512, num_classes=14)
        self.conformal_scorer = LearnableConformalScorer(vision_dim=768, text_dim=512)
    
    def forward(self, images, token_ids):
        vision_feat = self.vision_encoder(images)
        text_feat = self.text_encoder(token_ids)
        
        combined = torch.cat([vision_feat, text_feat], dim=1)
        fused_feat = self.fusion(combined)
        
        logits, log_var = self.classifier(fused_feat)
        conformal_scores = self.conformal_scorer(vision_feat, text_feat, logits)
        
        return {
            'logits': logits,
            'log_var': log_var,
            'conformal_scores': conformal_scores,
            'vision_feat': vision_feat,
            'text_feat': text_feat,
            'fused_feat': fused_feat
        }


class NoUncertaintyModel(nn.Module):
    """Ablation 5: No uncertainty head (deterministic)"""
    
    def __init__(self, vocab_size):
        super().__init__()
        self.vision_encoder = VisionEncoder(embed_dim=768)
        self.text_encoder = TextEncoder(vocab_size=vocab_size, embed_dim=512, num_layers=3)
        self.fusion = CrossModalFusion(vision_dim=768, text_dim=512, shared_dim=512)
        
        # Single classification head (no uncertainty)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 14)
        )
        
        self.conformal_scorer = LearnableConformalScorer(vision_dim=768, text_dim=512)
    
    def forward(self, images, token_ids):
        vision_feat = self.vision_encoder(images)
        text_feat = self.text_encoder(token_ids)
        fused_feat = self.fusion(vision_feat, text_feat)
        
        logits = self.classifier(fused_feat)
        log_var = torch.zeros_like(logits)  # No learned uncertainty
        
        conformal_scores = self.conformal_scorer(vision_feat, text_feat, logits)
        
        return {
            'logits': logits,
            'log_var': log_var,
            'conformal_scores': conformal_scores,
            'vision_feat': vision_feat,
            'text_feat': text_feat,
            'fused_feat': fused_feat
        }


class NoConformalModel(nn.Module):
    """Ablation 6: No conformal prediction (fixed prediction sets)"""
    
    def __init__(self, vocab_size):
        super().__init__()
        self.vision_encoder = VisionEncoder(embed_dim=768)
        self.text_encoder = TextEncoder(vocab_size=vocab_size, embed_dim=512, num_layers=3)
        self.fusion = CrossModalFusion(vision_dim=768, text_dim=512, shared_dim=512)
        self.classifier = UncertaintyAwareClassifier(input_dim=512, num_classes=14)
    
    def forward(self, images, token_ids):
        vision_feat = self.vision_encoder(images)
        text_feat = self.text_encoder(token_ids)
        fused_feat, _ = self.fusion(vision_feat, text_feat)  # Unpack tuple, ignore attention weights
        
        logits, log_var = self.classifier(fused_feat)
        conformal_scores = torch.zeros(images.size(0), device=images.device)  # No conformal
        
        return {
            'logits': logits,
            'log_var': log_var,
            'conformal_scores': conformal_scores,
            'vision_feat': vision_feat,
            'text_feat': text_feat,
            'fused_feat': fused_feat
        }


class NoContrastiveModel(nn.Module):
    """Ablation 7: No contrastive learning"""
    
    def __init__(self, vocab_size):
        super().__init__()
        # Same architecture as full model
        self.vision_encoder = VisionEncoder(embed_dim=768)
        self.text_encoder = TextEncoder(vocab_size=vocab_size, embed_dim=512, num_layers=3)
        self.fusion = CrossModalFusion(vision_dim=768, text_dim=512, shared_dim=512)
        self.classifier = UncertaintyAwareClassifier(input_dim=512, num_classes=14)
        self.conformal_scorer = LearnableConformalScorer(vision_dim=768, text_dim=512)
    
    def forward(self, images, token_ids):
        vision_feat = self.vision_encoder(images)
        text_feat = self.text_encoder(token_ids)
        fused_feat, _ = self.fusion(vision_feat, text_feat)  # Unpack tuple, ignore attention weights
        
        logits, log_var = self.classifier(fused_feat)
        conformal_scores = self.conformal_scorer(vision_feat, text_feat, logits)
        
        return {
            'logits': logits,
            'log_var': log_var,
            'conformal_scores': conformal_scores,
            'vision_feat': vision_feat,
            'text_feat': text_feat,
            'fused_feat': fused_feat
        }


class NoFocalLossModel(nn.Module):
    """Ablation 8: Standard BCE instead of focal loss"""
    
    def __init__(self, vocab_size):
        super().__init__()
        # Same architecture
        self.vision_encoder = VisionEncoder(embed_dim=768)
        self.text_encoder = TextEncoder(vocab_size=vocab_size, embed_dim=512, num_layers=3)
        self.fusion = CrossModalFusion(vision_dim=768, text_dim=512, shared_dim=512)
        self.classifier = UncertaintyAwareClassifier(input_dim=512, num_classes=14)
        self.conformal_scorer = LearnableConformalScorer(vision_dim=768, text_dim=512)
    
    def forward(self, images, token_ids):
        vision_feat = self.vision_encoder(images)
        text_feat = self.text_encoder(token_ids)
        fused_feat, _ = self.fusion(vision_feat, text_feat)  # Unpack tuple, ignore attention weights
        
        logits, log_var = self.classifier(fused_feat)
        conformal_scores = self.conformal_scorer(vision_feat, text_feat, logits)
        
        return {
            'logits': logits,
            'log_var': log_var,
            'conformal_scores': conformal_scores,
            'vision_feat': vision_feat,
            'text_feat': text_feat,
            'fused_feat': fused_feat
        }


class NoClassWeightsModel(nn.Module):
    """Ablation 9: No class weighting"""
    
    def __init__(self, vocab_size):
        super().__init__()
        self.vision_encoder = VisionEncoder(embed_dim=768)
        self.text_encoder = TextEncoder(vocab_size=vocab_size, embed_dim=512, num_layers=3)
        self.fusion = CrossModalFusion(vision_dim=768, text_dim=512, shared_dim=512)
        self.classifier = UncertaintyAwareClassifier(input_dim=512, num_classes=14)
        self.conformal_scorer = LearnableConformalScorer(vision_dim=768, text_dim=512)
    
    def forward(self, images, token_ids):
        vision_feat = self.vision_encoder(images)
        text_feat = self.text_encoder(token_ids)
        fused_feat, _ = self.fusion(vision_feat, text_feat)  # Unpack tuple, ignore attention weights
        
        logits, log_var = self.classifier(fused_feat)
        conformal_scores = self.conformal_scorer(vision_feat, text_feat, logits)
        
        return {
            'logits': logits,
            'log_var': log_var,
            'conformal_scores': conformal_scores,
            'vision_feat': vision_feat,
            'text_feat': text_feat,
            'fused_feat': fused_feat
        }