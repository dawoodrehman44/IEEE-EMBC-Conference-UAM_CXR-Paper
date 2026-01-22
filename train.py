def train_epoch(model, dataloader, optimizer, criterion, device, tokenizer, quantile=None):
    """Train for one epoch with all new losses"""
    model.train()
    
    total_losses = {
        'total': 0,
        'classification': 0,
        'contrastive': 0,
        'conformal': 0,
        'uncertainty_reg': 0
    }
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Training")
    for images, labels, findings, _ in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        token_ids = torch.stack([tokenizer.encode(f) for f in findings]).to(device)
        
        # Forward
        optimizer.zero_grad()
        outputs = model(images, token_ids)
        
        # Compute all losses
        loss, loss_dict = criterion(outputs, labels, quantile)
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Track
        for k, v in loss_dict.items():
            total_losses[k] += v
        
        preds = torch.sigmoid(outputs['logits']).detach().cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f"{loss_dict['total']:.4f}"})
    
    # Compute metrics
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    avg_losses = {k: v / len(dataloader) for k, v in total_losses.items()}
    auc = compute_auc(all_labels, all_preds)
    acc = accuracy_score(all_labels.flatten() > 0.5, all_preds.flatten() > 0.5)
    
    return avg_losses, auc, acc

def compute_auc(labels, preds):
    """Compute mean AUC"""
    aucs = []
    for i in range(labels.shape[1]):
        try:
            auc = roc_auc_score(labels[:, i], preds[:, i])
            aucs.append(auc)
        except:
            pass
    return np.mean(aucs) if aucs else 0.0


# ============================================================================
# CONFORMAL CALIBRATION (Now uses learned scores!)
# ============================================================================
def calibrate_conformal_quantile(model, dataloader, device, tokenizer, alpha=0.15, prev_quantile=None):
    """
    Calibrate with smoothing to prevent collapse
    """
    model.eval()
    all_scores = []
    
    with torch.no_grad():
        for images, labels, findings, _ in tqdm(dataloader, desc="Calibrating", leave=False):
            images = images.to(device)
            token_ids = torch.stack([tokenizer.encode(f) for f in findings]).to(device)
            
            outputs = model(images, token_ids)
            scores = outputs['conformal_scores']
            
            all_scores.append(scores.cpu())
    
    all_scores = torch.cat(all_scores)
    
    # Compute quantile
    n = len(all_scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    new_quantile = torch.quantile(all_scores, q_level)
    
    # ADD THIS: Smooth update with exponential moving average
    if prev_quantile is not None:
        # 70% old, 30% new - prevents sudden drops
        quantile = 0.7 * prev_quantile + 0.3 * new_quantile
        
        # Enforce minimum threshold
        quantile = max(quantile, 0.01)  # Never go below 0.01
    else:
        quantile = new_quantile
    
    print(f"Conformal quantile (LEARNED): {quantile:.4f} (raw: {new_quantile:.4f})")
    
    return torch.tensor(quantile)