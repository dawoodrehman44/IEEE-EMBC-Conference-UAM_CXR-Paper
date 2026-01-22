class EnhancedMetricsTracker:
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_auc': [], 'val_auc': [],
            'train_acc': [], 'val_acc': [],
            'ece': [], 'brier_score': [],
            'aleatoric_unc': [], 'epistemic_unc': [],
            'conformal_quantile': [],
            'coverage': [],          # ADD THIS
            'avg_set_size': []       # ADD THIS
        }
    
    def update(self, epoch, metrics):
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)
    
    def plot_metrics(self, epoch):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        epochs = list(range(1, epoch + 1))
        
        # Loss
        axes[0, 0].plot(epochs, self.history['train_loss'], label='Train', marker='o')
        axes[0, 0].plot(epochs, self.history['val_loss'], label='Val', marker='s')
        axes[0, 0].set_title('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # AUC
        axes[0, 1].plot(epochs, self.history['train_auc'], label='Train', marker='o')
        axes[0, 1].plot(epochs, self.history['val_auc'], label='Val', marker='s')
        axes[0, 1].set_title('AUC-ROC')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # ECE & Brier
        if self.history['ece']:
            axes[0, 2].plot(epochs, self.history['ece'], label='ECE', marker='o')
            axes[0, 2].plot(epochs, self.history['brier_score'], label='Brier', marker='s')
            axes[0, 2].set_title('Calibration Metrics')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1, 0].plot(epochs, self.history['train_acc'], label='Train', marker='o')
        axes[1, 0].plot(epochs, self.history['val_acc'], label='Val', marker='s')
        axes[1, 0].set_title('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Uncertainty
        if self.history['aleatoric_unc']:
            axes[1, 1].plot(epochs, self.history['aleatoric_unc'], label='Aleatoric', marker='o')
            axes[1, 1].plot(epochs, self.history['epistemic_unc'], label='Epistemic', marker='s')
            axes[1, 1].set_title('Uncertainty Components')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # Conformal quantile
        if self.history['conformal_quantile']:
            axes[1, 2].plot(epochs, self.history['conformal_quantile'], marker='o', color='purple')
            axes[1, 2].set_title('Learned Conformal Quantile')
            axes[1, 2].grid(True, alpha=0.3)

        # Coverage vs Set Size (replace bottom-right plot)
        if self.history['coverage']:
            ax1 = axes[1, 2]
            ax2 = ax1.twinx()
            
            ax1.plot(epochs, self.history['coverage'], label='Coverage', marker='o', color='green')
            ax1.axhline(y=0.85, color='r', linestyle='--', label='Target (0.85)')
            ax2.plot(epochs, self.history['avg_set_size'], label='Avg Set Size', marker='s', color='orange')
            
            ax1.set_title('Coverage vs Set Size')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Coverage', color='green')
            ax2.set_ylabel('Avg Set Size', color='orange')
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper right')
            ax1.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.checkpoint_dir, 'plots', 'training_metrics_enhanced.png'), dpi=150)
        plt.close()
    
    def save_history(self):
        # Convert numpy types to Python types
        history_serializable = {}
        for key, values in self.history.items():
            history_serializable[key] = [float(v) if isinstance(v, (np.float32, np.float64)) else v 
                                        for v in values]
        
        with open(os.path.join(self.checkpoint_dir, 'metrics_history_enhanced.json'), 'w') as f:
            json.dump(history_serializable, f, indent=2)