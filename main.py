def main():
    # Load data
    print("Loading data...")
    train_csv = os.path.join(config.cleaned_data_dir, "mimic_clean_train.csv")
    valid_csv = os.path.join(config.cleaned_data_dir, "mimic_clean_valid.csv")
    
    train_df = pd.read_csv(train_csv)
    valid_df = pd.read_csv(valid_csv)
    
    print(f"Train: {len(train_df)} samples")
    print(f"Valid: {len(valid_df)} samples\n")
    
    # ===== ADD THIS BLOCK: COMPUTE CLASS WEIGHTS =====
    print("Computing class weights for imbalanced diseases...")
    pos_counts = []
    for col in config.label_cols:
        # Handle -1 (uncertain) as positive
        pos_count = ((train_df[col] == 1) | (train_df[col] == -1)).sum()
        pos_counts.append(pos_count)
    
    pos_counts = np.array(pos_counts)
    total_samples = len(train_df)
    
    # Inverse frequency weighting
    class_weights = total_samples / (2 * pos_counts + 1e-6)  # Add epsilon to avoid division by zero
    
    # Normalize weights to [0.5, 2.0] range to avoid extreme values
    class_weights = np.clip(class_weights, 0.5, 2.0)
    
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(config.device)
    
    print("\nClass Weights:")
    for i, (disease, weight) in enumerate(zip(config.label_cols, class_weights)):
        print(f"  {disease:30s}: {weight:.3f} (samples: {pos_counts[i]})")
    print()
    # ===== END OF CLASS WEIGHTS BLOCK =====
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((config.img_size, config.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    valid_transform = transforms.Compose([
        transforms.Resize((config.img_size, config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Datasets
    train_dataset = MIMICCXRDataset(train_df, train_transform, config.image_root)
    valid_dataset = MIMICCXRDataset(valid_df, valid_transform, config.image_root)
    
    # Tokenizer
    tokenizer_path = os.path.join(config.checkpoint_dir, "tokenizer.pt")
    if os.path.exists(tokenizer_path):
        tokenizer = SimpleTokenizer.load(tokenizer_path)
    else:
        tokenizer = SimpleTokenizer(max_length=128)
        tokenizer.build_vocab(train_df['Findings_Clean'].dropna().tolist())
        tokenizer.save(tokenizer_path)
    
    # Split for calibration
    cal_size = int(len(train_dataset) * config.calibration_split)
    train_size = len(train_dataset) - cal_size
    train_subset, cal_subset = torch.utils.data.random_split(train_dataset, [train_size, cal_size])
    
    def collate_fn(batch):
        images, labels, findings, indices = zip(*batch)
        return torch.stack(images), torch.stack(labels), list(findings), list(indices)
    
    train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=8, pin_memory=True)
    cal_loader = DataLoader(cal_subset, batch_size=config.batch_size, shuffle=False,
                           collate_fn=collate_fn, num_workers=8, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=8, pin_memory=True)
    
    # Model
    model = UAM_CXR_Enhanced(vocab_size=len(tokenizer.vocab)).to(config.device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M\n")
    
    # Optimizer & Criterion
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = EnhancedLoss(class_weights=class_weights_tensor)  # NEW: Pass weights
    
    # Metrics
    metrics_tracker = EnhancedMetricsTracker(config.checkpoint_dir)
    
    # Training loop
    print("="*80)
    print("STARTING ENHANCED TRAINING")
    print("="*80 + "\n")
    
    # Initialize quantile tracker
    quantile = None
    prev_quantile = None
    
    for epoch in range(1, config.total_epochs + 1):
        print(f"\nEpoch {epoch}/{config.total_epochs}")
        print("-" * 80)
        
        # Train
        train_losses, train_auc, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, config.device, tokenizer, quantile
        )
        
        print(f"Train - Loss: {train_losses['total']:.4f}, AUC: {train_auc:.4f}, Acc: {train_acc:.4f}")
        print(f"  └─ Cls: {train_losses['classification']:.4f}, Contrast: {train_losses['contrastive']:.4f}, "
              f"Conformal: {train_losses['conformal']:.4f}, UncReg: {train_losses['uncertainty_reg']:.4f}")
        
        # Calibrate conformal quantile with smoothing
        if epoch % 5 == 0 or epoch == 1:
            quantile = calibrate_conformal_quantile(
                model, cal_loader, config.device, tokenizer, 
                config.conformal_alpha, prev_quantile  # ADD prev_quantile
            )
            prev_quantile = quantile  # Store for next calibration
        
        # Validate
        val_losses, val_auc, val_acc, unc_metrics, coverage, avg_set_size = validate_epoch(
            model, valid_loader, criterion, config.device, tokenizer, quantile
        )
        # Then print:
        if coverage is not None:
            print(f"Conformal - Coverage: {coverage:.4f}, Avg Set Size: {avg_set_size:.2f}")
        print(f"Valid - Loss: {val_losses['total']:.4f}, AUC: {val_auc:.4f}, Acc: {val_acc:.4f}")
        print(f"Uncertainty - ECE: {unc_metrics['ece']:.4f}, Brier: {unc_metrics['brier_score']:.4f}")
        print(f"             Aleatoric: {unc_metrics['aleatoric_uncertainty']:.4f}, "
              f"Epistemic: {unc_metrics['epistemic_uncertainty']:.4f}")
        
        # Update metrics
        metrics_tracker.update(epoch, {
            'train_loss': train_losses['total'],
            'val_loss': val_losses['total'],
            'train_auc': train_auc,
            'val_auc': val_auc,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'ece': unc_metrics['ece'],
            'brier_score': unc_metrics['brier_score'],
            'aleatoric_unc': unc_metrics['aleatoric_uncertainty'],
            'epistemic_unc': unc_metrics['epistemic_uncertainty'],
            'conformal_quantile': quantile.item() if quantile is not None else 0,
            'coverage': coverage if coverage is not None else 0,        # ADD THIS
            'avg_set_size': avg_set_size if avg_set_size is not None else 0  # ADD THIS
        })
        
        metrics_tracker.plot_metrics(epoch)
        metrics_tracker.save_history()
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics_history': metrics_tracker.history,
            'config': vars(config),
            'quantile': quantile
        }
        torch.save(checkpoint, os.path.join(config.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt'))
        
        if val_auc == max(metrics_tracker.history['coverage']):
            torch.save(checkpoint, os.path.join(config.checkpoint_dir, 'best_model_enhanced.pt'))
            print("✓ Best model saved!")
        
        print("-" * 80)
    
    print("\n" + "="*80)
    print("ENHANCED TRAINING COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()