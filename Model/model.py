# ============================================================================
# ENHANCED MODEL ARCHITECTURE
# ============================================================================
class VisionEncoder(nn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()
        from torchvision.models import resnet50, ResNet50_Weights
        
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection = nn.Linear(2048, embed_dim)
        
    def forward(self, x):
        features = self.backbone(x)
        features = features.squeeze(-1).squeeze(-1)
        return self.projection(features)

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, num_layers=3, num_heads=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=2048,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, token_ids):
        x = self.embedding(token_ids)
        x = self.transformer(x)
        return x.mean(dim=1)

class CrossModalFusion(nn.Module):
    def __init__(self, vision_dim, text_dim, shared_dim):
        super().__init__()
        self.vision_proj = nn.Linear(vision_dim, shared_dim)
        self.text_proj = nn.Linear(text_dim, shared_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=shared_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(shared_dim)
        
    def forward(self, vision_feat, text_feat):
        v = self.vision_proj(vision_feat).unsqueeze(1)
        t = self.text_proj(text_feat).unsqueeze(1)
        
        fused, attn_weights = self.attention(v, t, t)
        fused = fused.squeeze(1)
        
        return self.norm(fused + v.squeeze(1)), attn_weights


# ============================================================================
# NEW: LEARNABLE CONFORMAL SCORING NETWORK
# ============================================================================
class LearnableConformalScorer(nn.Module):
    """
    NOVEL CONTRIBUTION 1: Learnable Conformal Scoring
    
    Instead of post-hoc score computation, we LEARN what makes
    a prediction non-conformal using a neural network.
    """
    
    def __init__(self, vision_dim=768, text_dim=512, hidden_dim=256):
        super().__init__()
        
        # Score network that learns to predict non-conformity
        self.score_network = nn.Sequential(
            nn.Linear(vision_dim + text_dim + config.num_classes, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Score in [0, 1]
        )
        
        # Learnable temperature for score calibration
        self.temperature = nn.Parameter(torch.ones(1))
        self.threshold_high = nn.Parameter(torch.tensor(0.8))   # For n=5
        self.threshold_med = nn.Parameter(torch.tensor(0.5))    # For n=3
        
    def forward(self, vision_feat, text_feat, logits):
        """
        Learn non-conformity score from features + predictions
        
        Returns:
            score: [batch_size] - learned non-conformity scores
        """
        # Concatenate all available information
        combined = torch.cat([
            vision_feat,      # What image shows
            text_feat,        # What report says
            torch.sigmoid(logits)  # Model predictions
        ], dim=1)
        
        # Learn the score
        raw_score = self.score_network(combined).squeeze(-1)
        
        # Temperature-scaled score
        score = raw_score / (self.temperature + 1e-8)
        
        return score


# ============================================================================
# NEW: UNCERTAINTY-AWARE CLASSIFIER
# ============================================================================
class UncertaintyAwareClassifier(nn.Module):
    """
    NOVEL CONTRIBUTION 2: Uncertainty-Aware Prediction
    
    Predicts both:
    - Disease probabilities (epistemic uncertainty)
    - Aleatoric uncertainty (data noise)
    """
    
    def __init__(self, input_dim=512, num_classes=14):
        super().__init__()
        
        # Shared features
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )

        # Classification head (logits)
        self.classifier = nn.Linear(256, num_classes)

        # Aleatoric uncertainty head (log variance)
        self.uncertainty_head = nn.Linear(256, num_classes)

        # Initialize uncertainty head (MOVE HERE - AFTER creation)
        nn.init.constant_(self.uncertainty_head.bias, 0.0)
        nn.init.normal_(self.uncertainty_head.weight, 0, 0.001)
        
    def forward(self, x):
        features = self.shared(x)
        logits = self.classifier(features)
        log_var = self.uncertainty_head(features)
        log_var = torch.clamp(log_var, -5, 5)  # ADD THIS LINE
        return logits, log_var


# ============================================================================
# NEW: ENHANCED UAM-CXR MODEL
# ============================================================================
class UAM_CXR_Enhanced(nn.Module):
    """
    Enhanced UAM-CXR with:
    1. Learnable conformal scoring
    2. Uncertainty-aware predictions
    3. Contrastive vision-text alignment
    """
    
    def __init__(self, vocab_size):
        super().__init__()
        
        # Encoders
        self.vision_encoder = VisionEncoder(embed_dim=config.vision_embed_dim)
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            embed_dim=config.text_embed_dim,
            num_layers=3
        )
        
        # Fusion
        self.fusion = CrossModalFusion(
            vision_dim=config.vision_embed_dim,
            text_dim=config.text_embed_dim,
            shared_dim=config.shared_embed_dim
        )
        
        # NEW: Uncertainty-aware classifier
        self.classifier = UncertaintyAwareClassifier(
            input_dim=config.shared_embed_dim,
            num_classes=config.num_classes
        )
        
        # NEW: Learnable conformal scorer
        self.conformal_scorer = LearnableConformalScorer(
            vision_dim=config.vision_embed_dim,
            text_dim=config.text_embed_dim
        )
        
        # NEW: Projection heads for contrastive learning
        self.vision_proj_contrast = nn.Sequential(
            nn.Linear(config.vision_embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.text_proj_contrast = nn.Sequential(
            nn.Linear(config.text_embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
    
    def forward(self, images, token_ids):
        """
        Returns:
            logits: Disease predictions
            log_var: Aleatoric uncertainty
            conformal_scores: Learned non-conformity scores
            vision_feat, text_feat, fused_feat: For contrastive loss
        """
        # Encode modalities
        vision_feat = self.vision_encoder(images)
        text_feat = self.text_encoder(token_ids)
        
        # Fuse
        fused_feat, attn_weights = self.fusion(vision_feat, text_feat)
        
        # Predict with uncertainty
        logits, log_var = self.classifier(fused_feat)
        
        # Compute learned conformal scores
        conformal_scores = self.conformal_scorer(vision_feat, text_feat, logits)
        
        return {
            'logits': logits,
            'log_var': log_var,
            'conformal_scores': conformal_scores,
            'vision_feat': vision_feat,
            'text_feat': text_feat,
            'fused_feat': fused_feat,
            'attn_weights': attn_weights
        }
    
    def forward_with_dropout(self, images, token_ids, n_samples=10):
        """
        Monte Carlo Dropout for epistemic uncertainty
        
        Returns:
            mean_probs: [B, num_classes]
            epistemic_uncertainty: [B, num_classes]
        """
        self.train()  # Enable dropout
        
        predictions = []
        for _ in range(n_samples):
            outputs = self.forward(images, token_ids)
            probs = torch.sigmoid(outputs['logits'])
            predictions.append(probs)
        
        predictions = torch.stack(predictions)  # [n_samples, B, num_classes]
        
        mean_probs = predictions.mean(dim=0)
        epistemic_uncertainty = predictions.std(dim=0)
        
        return mean_probs, epistemic_uncertainty
