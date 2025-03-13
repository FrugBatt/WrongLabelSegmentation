import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=1.0, dropout=0.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA layers (low-rank decomposition)
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        
        # Initialize LoRA layers
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x):
        return self.dropout(self.lora_B(self.lora_A(x))) * self.scaling

class LoRAWrapper(nn.Module):
    def __init__(self, base_model, rank=8, alpha=1.0, dropout=0.0):
        super().__init__()
        self.base_model = base_model
        self.lora_layers = nn.ModuleDict()
        
        for name, module in base_model.named_modules():
            if isinstance(module, nn.Linear):
                self.lora_layers[name] = LoRALayer(module.in_features, module.out_features, rank, alpha, dropout)
                module.register_forward_hook(self._apply_lora_hook(name))
    
    def _apply_lora_hook(self, name):
        def hook(_, input, output):
            return output + self.lora_layers[name](input[0])
        return hook
    
    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)
    
    def train_lora_only(self):
        for param in self.base_model.parameters():
            param.requires_grad = False
        for param in self.lora_layers.parameters():
            param.requires_grad = True
