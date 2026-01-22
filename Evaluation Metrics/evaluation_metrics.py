def compute_ece(probs, labels, n_bins=10):
    """
    Expected Calibration Error (ECE)
    
    Measures: Are predicted probabilities well-calibrated?
    Lower is better.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Samples in this bin
        in_bin = (probs >= bin_lower) & (probs < bin_upper)
        
        if in_bin.sum() > 0:
            bin_acc = labels[in_bin].mean()
            bin_conf = probs[in_bin].mean()
            bin_weight = in_bin.sum() / len(probs)
            
            ece += bin_weight * np.abs(bin_acc - bin_conf)
    
    return ece

def compute_brier_score(probs, labels):
    """
    Brier Score
    
    Measures: Mean squared error between probabilities and labels
    Lower is better.
    """
    return ((probs - labels) ** 2).mean()

def compute_uncertainty_metrics(model, dataloader, device, tokenizer):
    """
    Compute comprehensive uncertainty metrics:
    - ECE (calibration)
    - Brier score
    - Aleatoric uncertainty (average)
    - Epistemic uncertainty (MC dropout)
    """
    model.eval()
    
    all_probs = []
    all_labels = []
    all_aleatoric = []
    all_epistemic = []
    
    with torch.no_grad():
        for images, labels, findings, _ in tqdm(dataloader, desc="Computing metrics", leave=False):
            images = images.to(device)
            labels_np = labels.numpy()
            
            token_ids = torch.stack([tokenizer.encode(f) for f in findings]).to(device)
            
            # Standard forward pass
            outputs = model(images, token_ids)
            probs = torch.sigmoid(outputs['logits']).cpu().numpy()
            
            # Aleatoric uncertainty
            aleatoric = torch.exp(outputs['log_var']).cpu().numpy()
            
            # Epistemic uncertainty (MC dropout)
            mean_probs, epistemic = model.forward_with_dropout(
                images, token_ids, n_samples=config.mc_samples
            )
            epistemic = epistemic.cpu().numpy()
            
            all_probs.append(probs)
            all_labels.append(labels_np)
            all_aleatoric.append(aleatoric)
            all_epistemic.append(epistemic)
    
    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)
    all_aleatoric = np.vstack(all_aleatoric)
    all_epistemic = np.vstack(all_epistemic)
    
    # Compute metrics
    ece = compute_ece(all_probs.flatten(), all_labels.flatten())
    brier = compute_brier_score(all_probs.flatten(), all_labels.flatten())
    avg_aleatoric = all_aleatoric.mean()
    avg_epistemic = all_epistemic.mean()
    
    return {
        'ece': ece,
        'brier_score': brier,
        'aleatoric_uncertainty': avg_aleatoric,
        'epistemic_uncertainty': avg_epistemic
    }
