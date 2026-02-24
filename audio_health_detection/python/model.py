#!/usr/bin/env python3
"""
Cough Classification Model for Canine Health Detection

This module implements the CNN14 neural network architecture for classifying
dog coughs into health categories (healthy, kennel cough, heart failure cough).

ARCHITECTURE NOTE:
==================
This model EXACTLY matches the CNN14 architecture from PANNs (Pre-trained Audio
Neural Networks) so we can load the pretrained weights correctly.

PANNs CNN14 structure:
- bn0: BatchNorm2d on input (64 features for 64 mel bins)
- conv_block1: 1 → 64 channels (takes raw spectrogram)
- conv_block2: 64 → 128
- conv_block3: 128 → 256  
- conv_block4: 256 → 512
- conv_block5: 512 → 1024
- conv_block6: 1024 → 2048
- fc1: 2048 → 2048 (with ReLU)
- fc_audioset: 2048 → 527 (original AudioSet classes - we replace this)

We load all layers EXCEPT fc_audioset, then add our own classification head.

DEEP LEARNING CONCEPTS EXPLAINED:
=================================

1. CONVOLUTIONAL NEURAL NETWORKS (CNNs)
   - Originally designed for images, work great on spectrograms too!
   - "Convolution" = sliding a small filter across the input, detecting patterns
   - Early layers detect edges/textures, later layers detect complex shapes
   - Spectrograms ARE images: height=frequency, width=time, brightness=intensity

2. WHY CNN14 SPECIFICALLY?
   - From PANNs (Pre-trained Audio Neural Networks) project
   - Pre-trained on AudioSet: 5000+ hours, 527 sound classes
   - Already knows "what sound looks like" - we just fine-tune for dog coughs
   - Good accuracy/speed tradeoff (80M parameters, runs on consumer GPUs)

3. TRANSFER LEARNING
   - Instead of training from scratch (needs millions of samples)
   - We take a model trained on related task (general audio)
   - Freeze early layers (keep general audio knowledge)
   - Train only final layers for our specific task (dog cough types)
   - Works with just hundreds of samples instead of millions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def init_layer(layer):
    """Initialize a Linear or Convolutional layer."""
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a BatchNorm layer."""
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock(nn.Module):
    """
    A convolutional block matching PANNs CNN14 architecture.
    
    Structure:
        Conv2d → BatchNorm → ReLU → Conv2d → BatchNorm → ReLU → AvgPool
    
    This "double conv" pattern (from VGG networks) is very effective:
    - Two 3x3 convolutions have the same receptive field as one 5x5
    - But with fewer parameters and more non-linearity
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.init_weight()
    
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)
    
    def forward(self, x: torch.Tensor, pool_size: Tuple[int, int] = (2, 2)):
        """Forward pass with configurable pooling size."""
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.avg_pool2d(x, kernel_size=pool_size)
        return x


class CoughClassifier(nn.Module):
    """
    CNN14-based classifier for dog cough detection.
    
    This network takes a mel spectrogram as input and outputs probabilities
    for each cough category. Architecture exactly matches PANNs CNN14 for
    compatibility with pretrained weights.
    
    Example usage:
        model = CoughClassifier(num_classes=4)
        mel_spec = torch.randn(1, 1, 64, 251)  # (batch, channel, freq, time)
        output = model(mel_spec)
        # output['logits'] shape: (1, 4)
        # output['probabilities'] shape: (1, 4)
    """
    
    # Class labels for reference
    LABELS = ['healthy', 'kennel_cough', 'heart_failure_cough', 'other_respiratory']
    
    def __init__(
        self,
        num_classes: int = 4,
        dropout: float = 0.5,
        pretrained_path: Optional[str] = None,
    ):
        """
        Initialize the cough classifier.
        
        Args:
            num_classes: Number of output classes (4 for our task)
            dropout: Dropout probability for regularization
            pretrained_path: Path to pretrained CNN14 weights (optional)
        """
        super().__init__()
        
        self.num_classes = num_classes
        
        # ====================================
        # CNN14 Backbone (matches PANNs exactly)
        # ====================================
        
        # Initial batch normalization on the mel spectrogram input
        # Applied across the frequency dimension (64 mel bins)
        self.bn0 = nn.BatchNorm2d(64)
        
        # Convolutional blocks - note conv_block1 takes 1 input channel
        # This is the raw spectrogram (single "grayscale" channel)
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)
        
        # Fully connected layer from PANNs (we keep this for transfer learning)
        self.fc1 = nn.Linear(2048, 2048, bias=True)
        
        # ====================================
        # Our Custom Classification Head
        # ====================================
        # Replace PANNs' 527-class AudioSet head with our 4-class head
        self.fc_cough = nn.Linear(2048, num_classes, bias=True)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.init_weight()
        
        # Load pretrained weights if provided
        if pretrained_path and Path(pretrained_path).exists():
            self._load_pretrained_weights(pretrained_path)
    
    def init_weight(self):
        """Initialize all weights."""
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_cough)
    
    def _load_pretrained_weights(self, path: str) -> None:
        """
        Load pretrained CNN14 weights from PANNs.
        
        The pretrained model was trained on AudioSet (527 classes).
        We load all layers EXCEPT fc_audioset, which we replace with
        our own classification head (fc_cough).
        """
        print(f"Loading pretrained weights from {path}")
        
        try:
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            
            # Handle different checkpoint formats
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            # Get our model's state dict
            model_dict = self.state_dict()
            
            # Filter checkpoint: only load layers that exist in our model
            # Skip fc_audioset (PANNs output layer) - we use fc_cough instead
            pretrained_dict = {}
            skipped = []
            
            for key, value in state_dict.items():
                # Skip the AudioSet classification layer
                if 'fc_audioset' in key:
                    skipped.append(key)
                    continue
                
                # Check if this key exists in our model
                if key in model_dict:
                    # Check shape matches
                    if value.shape == model_dict[key].shape:
                        pretrained_dict[key] = value
                    else:
                        skipped.append(f"{key} (shape mismatch: {value.shape} vs {model_dict[key].shape})")
                else:
                    skipped.append(f"{key} (not in model)")
            
            # Load the filtered weights
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
            
            print(f"  ✓ Loaded {len(pretrained_dict)} weight tensors")
            print(f"  ✓ Skipped {len(skipped)} tensors (AudioSet head + mismatches)")
            
            if len(pretrained_dict) > 0:
                print("  ✓ Pretrained backbone loaded successfully!")
            
        except Exception as e:
            print(f"  Warning: Could not load pretrained weights: {e}")
            print("  Continuing with random initialization")
    
    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input mel spectrogram, shape (batch, 1, n_mels, time_frames)
               Typical shape: (batch, 1, 64, 251) for 5 seconds of audio
            return_embedding: If True, also return the 2048-dim embedding
            
        Returns:
            Dictionary containing:
            - 'logits': Raw scores for each class (batch, num_classes)
            - 'probabilities': Softmax probabilities (batch, num_classes)
            - 'embedding': (optional) 2048-dim feature vector (batch, 2048)
        """
        # Input shape: (batch, 1, freq, time) e.g., (16, 1, 64, 251)
        
        # Transpose for batch norm: (batch, 1, freq, time) → (batch, freq, 1, time)
        # Then apply BN across freq dimension, then transpose back
        x = x.transpose(1, 2)  # (batch, freq, 1, time)
        x = self.bn0(x)
        x = x.transpose(1, 2)  # (batch, 1, freq, time)
        
        # Pass through conv blocks with pooling
        # Each block halves spatial dimensions via 2x2 avg pooling
        x = self.conv_block1(x, pool_size=(2, 2))  # → (batch, 64, freq/2, time/2)
        x = self.dropout(x)
        x = self.conv_block2(x, pool_size=(2, 2))  # → (batch, 128, freq/4, time/4)
        x = self.dropout(x)
        x = self.conv_block3(x, pool_size=(2, 2))  # → (batch, 256, freq/8, time/8)
        x = self.dropout(x)
        x = self.conv_block4(x, pool_size=(2, 2))  # → (batch, 512, freq/16, time/16)
        x = self.dropout(x)
        x = self.conv_block5(x, pool_size=(2, 2))  # → (batch, 1024, freq/32, time/32)
        x = self.dropout(x)
        x = self.conv_block6(x, pool_size=(1, 1))  # → (batch, 2048, freq/32, time/32) no pooling
        x = self.dropout(x)
        
        # Global pooling: (batch, 2048, H, W) → (batch, 2048)
        # Combine mean and max pooling for richer features
        x_mean = torch.mean(x, dim=(2, 3))  # Global average pooling
        x_max, _ = torch.max(x, dim=2)       # Max over frequency
        x_max, _ = torch.max(x_max, dim=2)   # Max over time
        x = x_mean + x_max  # Combine both
        
        # Fully connected layer (from pretrained)
        x = self.dropout(x)
        x = F.relu_(self.fc1(x))
        
        embedding = x
        
        # Our classification head
        x = self.dropout(x)
        logits = self.fc_cough(x)
        
        # Compute probabilities
        probabilities = F.softmax(logits, dim=1)
        
        result = {
            'logits': logits,
            'probabilities': probabilities,
        }
        
        if return_embedding:
            result['embedding'] = embedding
        
        return result
    
    def predict(
        self,
        x: torch.Tensor,
    ) -> Dict[str, any]:
        """
        Make a prediction with human-readable output.
        
        Args:
            x: Input mel spectrogram, shape (batch, 1, n_mels, time_frames)
            
        Returns:
            Dictionary with prediction results
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            probs = output['probabilities']
            
            # Get top prediction
            top_prob, top_idx = probs.max(dim=1)
            
            # Build results for each sample in batch
            results = []
            for i in range(probs.size(0)):
                result = {
                    'label': self.LABELS[top_idx[i].item()],
                    'confidence': top_prob[i].item(),
                    'all_probabilities': {
                        label: probs[i, j].item()
                        for j, label in enumerate(self.LABELS)
                    }
                }
                results.append(result)
            
            return results[0] if len(results) == 1 else results
    
    def save_checkpoint(
        self,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
        **kwargs,
    ) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'num_classes': self.num_classes,
            'epoch': epoch,
            'labels': self.LABELS,
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        checkpoint.update(kwargs)
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    @classmethod
    def load_checkpoint(
        cls,
        path: str,
        device: str = 'cpu',
    ) -> 'CoughClassifier':
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        
        num_classes = checkpoint.get('num_classes', 4)
        model = cls(num_classes=num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        print(f"Model loaded from {path}")
        print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
        
        return model


# Quick test when run directly
if __name__ == "__main__":
    print("=" * 60)
    print("Cough Classifier Model - Test")
    print("=" * 60)
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Create model
    print("\nCreating model...")
    model = CoughClassifier(num_classes=4)
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 4
    n_mels = 64
    time_frames = 251  # ~5 seconds at 16kHz with hop=320
    
    x = torch.randn(batch_size, 1, n_mels, time_frames).to(device)
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    output = model(x, return_embedding=True)
    
    print(f"Output logits shape: {output['logits'].shape}")
    print(f"Output probabilities shape: {output['probabilities'].shape}")
    print(f"Output embedding shape: {output['embedding'].shape}")
    
    # Test prediction
    print("\nTesting prediction...")
    prediction = model.predict(x[:1])
    print(f"Prediction: {prediction['label']}")
    print(f"Confidence: {prediction['confidence']:.2%}")
    print("All probabilities:")
    for label, prob in prediction['all_probabilities'].items():
        print(f"  {label}: {prob:.2%}")
    
    # Test with pretrained weights if available
    pretrained_path = Path("../models/pretrained/Cnn14_mAP=0.431.pth")
    if pretrained_path.exists():
        print("\n" + "-" * 40)
        print("Testing pretrained weight loading...")
        print("-" * 40)
        model_pretrained = CoughClassifier(
            num_classes=4,
            pretrained_path=str(pretrained_path)
        )
        model_pretrained.to(device)
        
        # Test that it still works
        output = model_pretrained(x)
        print(f"Output with pretrained: {output['logits'].shape}")
    
    print("\n✓ Model working correctly!")
