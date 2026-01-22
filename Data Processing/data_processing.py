# ============================================================================
# DATASET
# ============================================================================
class MIMICCXRDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, transform, image_root):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.image_root = image_root
        self.label_cols = config.label_cols
        
        for col in self.label_cols:
            if col in self.dataframe.columns:
                self.dataframe[col] = self.dataframe[col].fillna(0)
                self.dataframe[col] = self.dataframe[col].apply(
                    lambda x: 1 if x in [1.0, -1.0] else 0
                )
        
        print(f"Dataset: {len(self.dataframe)} samples")
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        
        img_path = os.path.join(self.image_root, row['Path'])
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except:
            image = torch.zeros(3, config.img_size, config.img_size)
        
        labels = torch.tensor(row[self.label_cols].values.astype(np.float32))
        findings = str(row['Findings_Clean']) if pd.notna(row['Findings_Clean']) else ""
        
        return image, labels, findings, idx

# ============================================================================
# TOKENIZER
# ============================================================================
class SimpleTokenizer:
    def __init__(self, max_length=128):
        self.max_length = max_length
        self.vocab = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
        self.idx_to_token = {v: k for k, v in self.vocab.items()}
    
    def build_vocab(self, texts):
        print("Building tokenizer vocabulary...")
        for text in tqdm(texts, desc="Processing texts"):
            tokens = text.lower().split()
            for token in tokens:
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)
        
        self.idx_to_token = {v: k for k, v in self.vocab.items()}
        print(f"Vocabulary size: {len(self.vocab)}")
    
    def encode(self, text):
        tokens = text.lower().split()[:self.max_length-2]
        ids = [self.vocab['<BOS>']]
        ids.extend([self.vocab.get(t, self.vocab['<UNK>']) for t in tokens])
        ids.append(self.vocab['<EOS>'])
        
        if len(ids) < self.max_length:
            ids.extend([self.vocab['<PAD>']] * (self.max_length - len(ids)))
        
        return torch.tensor(ids[:self.max_length])
        
    def decode_to_tokens(self, text):
        """Convert text to list of tokens"""
        # Lowercase and split
        tokens = text.lower().split()
        
        # Filter to vocab tokens only
        result = []
        for token in tokens:
            if token in self.vocab:
                result.append(token)
            else:
                result.append('<UNK>')
        
        return result

    def save(self, path):
        torch.save({'vocab': self.vocab, 'max_length': self.max_length}, path)
    
    @classmethod
    def load(cls, path):
        data = torch.load(path)
        tokenizer = cls(max_length=data['max_length'])
        tokenizer.vocab = data['vocab']
        tokenizer.idx_to_token = {v: k for k, v in tokenizer.vocab.items()}
        return tokenizer