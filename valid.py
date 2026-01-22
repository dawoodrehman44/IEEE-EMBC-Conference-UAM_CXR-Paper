def validate_epoch(model, dataloader, criterion, device, tokenizer, quantile=None):
    """Validate with uncertainty metrics"""
    model.eval()
    
    total_losses = {'total': 0, 'classification': 0, 'contrastive': 0, 'conformal': 0, 'uncertainty_reg': 0}
    all_preds = []
    all_labels = []
    all_pred_sets = []
    
    with torch.no_grad():
        for images, labels, findings, _ in tqdm(dataloader, desc="Validating", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            
            token_ids = torch.stack([tokenizer.encode(f) for f in findings]).to(device)
            
            outputs = model(images, token_ids)
            loss, loss_dict = criterion(outputs, labels, quantile)
            
            for k, v in loss_dict.items():
                total_losses[k] += v
            
            probs = torch.sigmoid(outputs['logits']).cpu().numpy()
            all_preds.append(probs)
            all_labels.append(labels.cpu().numpy())
            
            # ADD THIS: Compute prediction sets
            if quantile is not None:
                # Simple threshold-based sets using learned scores
                batch_pred_sets = []
                for i in range(len(images)):
                    score = outputs['conformal_scores'][i].item()
                    sorted_idx = np.argsort(probs[i])[::-1]  # Descending
                    
                    # CHANGED: Higher base set size (3 instead of 2)
                    # This ensures minimum 3 diseases included
                    n_include = max(3, min(5, int(3 * score / (quantile + 1e-8))))
                    
                    batch_pred_sets.append(sorted_idx[:n_include].tolist())
                
                all_pred_sets.extend(batch_pred_sets)
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    # Per-disease analysis
    print("\n=== PER-DISEASE PERFORMANCE ===")
    for i, disease in enumerate(config.label_cols):
        disease_labels = all_labels[:, i]
        disease_preds = all_preds[:, i]
        
        # Count positives
        n_pos = (disease_labels > 0.5).sum()
        
        # AUC
        try:
            disease_auc = roc_auc_score(disease_labels, disease_preds)
        except:
            disease_auc = 0.0
        
        # Accuracy
        disease_acc = ((disease_preds > 0.5) == (disease_labels > 0.5)).mean()
        
        print(f"{disease:30s} | AUC: {disease_auc:.3f} | Acc: {disease_acc:.3f} | Pos: {n_pos}")
    print("="*70)
    
    avg_losses = {k: v / len(dataloader) for k, v in total_losses.items()}
    auc = compute_auc(all_labels, all_preds)
    acc = accuracy_score(all_labels.flatten() > 0.5, all_preds.flatten() > 0.5)
    
    # Compute uncertainty metrics
    # Compute uncertainty metrics
    uncertainty_metrics = compute_uncertainty_metrics(model, dataloader, device, tokenizer)
    
    # ADD THIS BLOCK BEFORE RETURN:
    # Compute coverage
    coverage = None
    avg_set_size = None

    if all_pred_sets:
        covered = 0
        for i in range(len(all_labels)):
            true_labels = set(np.where(all_labels[i] > 0.5)[0])
            pred_set = set(all_pred_sets[i])
            if true_labels.issubset(pred_set):
                covered += 1
        coverage = covered / len(all_labels)
        avg_set_size = np.mean([len(s) for s in all_pred_sets])
        
        # ADD DIAGNOSTIC HERE (inside the if all_pred_sets block):
        print(f"\n=== COVERAGE DIAGNOSTIC ===")
        print(f"Total samples: {len(all_labels)}")
        
        missed_samples = 0
        missed_diseases = []
        avg_true_positives = []
        
        for i in range(len(all_labels)):
            true_labels = set(np.where(all_labels[i] > 0.5)[0])
            pred_set = set(all_pred_sets[i])
            
            avg_true_positives.append(len(true_labels))
            
            if not true_labels.issubset(pred_set):
                missed_samples += 1
                missed = true_labels - pred_set
                missed_diseases.extend(list(missed))
        
        print(f"Missed samples: {missed_samples} ({missed_samples/len(all_labels)*100:.1f}%)")
        print(f"Avg true positives per sample: {np.mean(avg_true_positives):.2f}")
        print(f"Avg set size: {avg_set_size:.2f}")
        print(f"Missed disease distribution:")
        for d in range(14):
            count = missed_diseases.count(d)
            if count > 0:
                print(f"  {config.label_cols[d]}: {count} times")
        print("="*30 + "\n")

    return avg_losses, auc, acc, uncertainty_metrics, coverage, avg_set_size
