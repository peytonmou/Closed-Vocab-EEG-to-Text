import torch
import torch.nn as nn
import torch.nn.init as init
from math import floor

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm: int = 1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(x)

class BaseModel(nn.Module):
    def __init__(self, num_classes, device='cuda', **kwargs):
        super(BaseModel, self).__init__()
        self.num_classes = num_classes
        self.device = device

# Weight initialization function for NiceEEG
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

class PatchEmbedding(nn.Module):
    def __init__(self, k=40, m1=25, m2=51, s=5, ch=63):
        super().__init__()
        # revised from shallownet
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, k, (1, m1), (1, 1)),
            nn.AvgPool2d((1, m2), (1, s)),
            nn.BatchNorm2d(k),
            nn.ELU(),
            nn.Conv2d(k, k, (ch, 1), (1, 1)),
            nn.BatchNorm2d(k),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(k, k, (1, 1), stride=(1, 1)),
            nn.Flatten(start_dim=2),  # Replace einops Rearrange
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tsconv(x)
        x = self.projection(x)
        x = x.transpose(1, 2)  # [b, seq_len, features]
        return x

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return x

class NiceEEG(BaseModel):
    def __init__(self, num_classes, device='cuda', **kwargs):
        super().__init__(num_classes, device=device, **kwargs)
        self.build_model(**kwargs)

    def build_model(self,
                    time_steps: int = 250,
                    num_electrodes: int = 64,
                    k: int = 40,
                    m1: int = 25,
                    m2: int = 51,
                    s: int = 5,
                    proj_dim: int = 768,
                    dropout: float = 0.5):
        
        self.time_steps = time_steps
        self.num_electrodes = num_electrodes
        self.k = k
        self.m1 = m1
        self.m2 = m2
        self.s = s
        self.proj_dim = proj_dim
        self.dropout = dropout

        self.Enc_eeg = nn.Sequential(
            PatchEmbedding(k=k, m1=m1, m2=m2, s=s, ch=self.num_electrodes),
            FlattenHead()
        ).apply(weights_init_normal)

        # Calculate the embedding dimension of EEG after EEG encoder
        eeg_embedding_dim = int(k * floor(((self.time_steps - m1 + 1) - m2) / s + 1))
        
        self.Proj_eeg = nn.Sequential(
            nn.Linear(eeg_embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(dropout),)),
            nn.LayerNorm(proj_dim),
        ).apply(weights_init_normal)

        # Classification head
        self.classifier = nn.Linear(proj_dim, self.num_classes).apply(weights_init_normal)

        # Optimizer and loss
        self.learning_rate = 1e-3
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        # x shape: [B, 1, C, T] - add channel dimension if needed
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dimension [B, 1, C, T]
        eeg_features = self.Enc_eeg(x)
        eeg_features = self.Proj_eeg(eeg_features)
        logits = self.classifier(eeg_features)
        return logits

class NetTraST(BaseModel):
    def __init__(self, num_classes, device='cuda', **kwargs):
        super().__init__(num_classes, device=device, **kwargs)
        self.build_model(**kwargs)

    def build_model(self,
                    time_steps: int = 250,
                    num_electrodes: int = 64,
                    embed_dim: int = 64,
                    kernel_num: int = 64,
                    kernel_size: int = 3,
                    nhead: int = 8,
                    dim_feedforward: int = 256,
                    num_layers: int = 3,
                    vocab_size: int = 128,
                    dropout: float = 0.5):
        
        self.time_steps = time_steps
        self.num_electrodes = num_electrodes
        self.embed_dim = embed_dim
        self.kernel_num = kernel_num
        self.kernel_size = kernel_size
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.dropout = dropout

        # Input shape: [batch, channels (64), time_points (250)]
        self.batch_norm1 = nn.BatchNorm1d(self.embed_dim)  # Normalize along channels
        
        # Spatial pathway (processing channels)
        self.spatial_conv = nn.Conv1d(
            in_channels=self.embed_dim, 
            out_channels=self.kernel_num, 
            kernel_size=self.kernel_size, 
            padding=self.kernel_size//2
        )
        self.spatial_tra = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.kernel_num,
                nhead=self.nhead,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                batch_first=False
            ),
            num_layers=3,
        )
        
        # Temporal pathway (processing time points)
        self.temporal_conv = nn.Conv1d(
            in_channels=self.vocab_size, 
            out_channels=self.kernel_num, 
            kernel_size=self.kernel_size, 
            padding=self.kernel_size//2
        )
        self.temporal_tra = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.kernel_num,
                nhead=self.nhead,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                batch_first=False
            ),
            num_layers=3,
        )
        
        # Main transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.kernel_num,
                nhead=self.nhead,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                batch_first=False
            ),
            num_layers=self.num_layers,
        )
        
        # Classification head
        self.batch_norm2 = nn.BatchNorm1d(self.kernel_num)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.kernel_num * self.vocab_size, self.kernel_num)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.fc2 = nn.Linear(self.kernel_num, self.num_classes)
        self.activation = nn.RReLU(0.1, 0.3)

        # Optimizer and loss
        self.learning_rate = 1e-3
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        # Input shape: [batch, 1, channels, time_points] -> reshape to [batch, channels, time_points]
        if x.dim() == 4:
            x = x.squeeze(1)  # Remove channel dimension [B, C, T]
        
        # Ensure dimensions match expected input
        if x.shape[1] != self.embed_dim:
            # Use first 'embed_dim' channels or interpolate
            x = x[:, :self.embed_dim, :]
        
        if x.shape[2] != self.vocab_size:
            # Interpolate time dimension to match vocab_size
            x = nn.functional.interpolate(x, size=self.vocab_size)
        
        x = self.batch_norm1(x)
        
        # Spatial pathway
        x1 = self.spatial_conv(x)  # [batch, kernel_num, vocab_size]
        x1 = x1.permute(2, 0, 1)  # [vocab_size, batch, kernel_num] for transformer
        x1 = self.spatial_tra(x1)
        x1 = x1.permute(1, 2, 0)  # [batch, kernel_num, vocab_size]
        
        # Temporal pathway
        x2 = x.permute(0, 2, 1)  # [batch, vocab_size, embed_dim]
        x2 = self.temporal_conv(x2)  # [batch, kernel_num, embed_dim]
        x2 = x2.permute(2, 0, 1)  # [embed_dim, batch, kernel_num] for transformer
        x2 = self.temporal_tra(x2)
        x2 = x2.permute(1, 2, 0)  # [batch, kernel_num, embed_dim]
        
        # Align dimensions for addition
        x2 = nn.functional.interpolate(x2, size=x1.shape[2])  # Upsample to match x1's time dimension
        
        # Combine pathways
        x = x1 + x2  # [batch, kernel_num, vocab_size]
        
        # Final processing
        x = x.permute(2, 0, 1)  # [vocab_size, batch, kernel_num]
        x = self.transformer(x)
        x = x.permute(1, 2, 0)  # [batch, kernel_num, vocab_size]
        x = self.batch_norm2(x)
        
        # Classification
        x = self.flatten(x)  # [batch, kernel_num * vocab_size]
        x = self.activation(self.fc1(x))
        x = self.dropout_layer(x)
        x = self.fc2(x)
        
        return x

class EEGNet(BaseModel):
    def __init__(self, num_classes, device='cuda', **kwargs):
        super().__init__(num_classes, device=device, **kwargs)
        self.build_model(**kwargs)

    def build_model(self,
                    time_steps: int = 250,
                    num_electrodes: int = 64,
                    F1: int = 8,
                    F2: int = 16,
                    D: int = 2,
                    kernel_1: int = 64,
                    kernel_2: int = 16,
                    dropout: float = 0.25,
                    momentum: float = 0.01,
                    learning_rate: float = 1e-3):
        
        self.chunk_size = time_steps
        self.num_electrodes = num_electrodes
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.kernel_1 = kernel_1
        self.kernel_2 = kernel_2
        self.dropout = dropout
        self.momentum = momentum
        self.learning_rate = learning_rate

        # Block 1: Temporal convolution and depthwise spatial convolution
        self.block1 = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, self.kernel_1), stride=1, padding=(0, self.kernel_1 // 2), bias=False),
            nn.BatchNorm2d(self.F1, momentum=self.momentum, affine=True, eps=1e-3),
            Conv2dWithConstraint(self.F1,
                                 self.F1 * self.D, (self.num_electrodes, 1),
                                 max_norm=1,
                                 stride=1,
                                 padding=(0, 0),
                                 groups=self.F1,
                                 bias=False),
            nn.BatchNorm2d(self.F1 * self.D, momentum=self.momentum, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, 4), stride=4),
            nn.Dropout(self.dropout)
        )

        # Block 2: Separable convolution
        self.block2 = nn.Sequential(
            nn.Conv2d(self.F1 * self.D,
                      self.F1 * self.D, (1, self.kernel_2),
                      stride=1,
                      padding=(0, self.kernel_2 // 2),
                      bias=False,
                      groups=self.F1 * self.D),
            nn.Conv2d(self.F1 * self.D, self.F2, 1, padding=(0, 0), groups=1, bias=False, stride=1),
            nn.BatchNorm2d(self.F2, momentum=self.momentum, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, 8), stride=8),
            nn.Dropout(self.dropout)
        )

        # Calculate feature dimension
        self.feature_dim = self._calculate_feature_dim()
        self.classifier = nn.Linear(self.feature_dim, self.num_classes)
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def _calculate_feature_dim(self):
        with torch.no_grad():
            mock_input = torch.zeros(1, 1, self.num_electrodes, self.chunk_size)
            mock_output = self.block1(mock_input)
            mock_output = self.block2(mock_output)
            return mock_output.view(1, -1).size(1)

    def forward(self, x):
        # x shape: [batch_size, 1, channels, time_points]
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def create_model(model_name, num_classes, **kwargs):
    """Factory function to create models"""
    model_classes = {
        'eegnet': EEGNet,
        'niceeeg': NiceEEG,
        'nettrast': NetTraST
    }
    
    if model_name.lower() not in model_classes:
        raise ValueError(f"Model {model_name} not supported. Choose from {list(model_classes.keys())}")
    
    return model_classes[model_name.lower()](num_classes, **kwargs)