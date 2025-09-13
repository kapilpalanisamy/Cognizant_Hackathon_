#!/usr/bin/env python3
"""
Convert model for ML API compatibility
This script loads the original model and saves just the state dict
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm
import sys
import os

class FinalModelDetector(nn.Module):
    """
    ⚡ Fast precision detector - optimized EfficientNet-B1 for speed
    """
    
    def __init__(self, num_classes=2):
        super(FinalModelDetector, self).__init__()
        
        # EfficientNet-B1 - good balance of speed and accuracy
        self.backbone = timm.create_model('efficientnet_b1', pretrained=True, num_classes=0)
        self.num_features = self.backbone.num_features
        
        # Streamlined classifier for speed
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self.num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
        # Precision-optimized bias initialization
        with torch.no_grad():
            self.classifier[-1].bias[0] = 1.0   # Moderate bias toward fraud
            self.classifier[-1].bias[1] = -0.5  # Moderate bias against non-fraud
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output

def convert_model():
    """Convert the original model to state dict format"""
    
    # Load the original model
    model_path = "../../final_model.pth"
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    try:
        print(f"📁 Loading original model from: {model_path}")
        
        # Try different loading approaches
        try:
            # First try: load as full model
            original_model = torch.load(model_path, map_location='cpu', weights_only=False)
            state_dict = original_model.state_dict()
            print("✅ Loaded as full model object")
        except Exception as e1:
            print(f"⚠️  Full model loading failed: {e1}")
            try:
                # Second try: load as state dict
                state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
                print("✅ Loaded as state dict")
            except Exception as e2:
                print(f"❌ State dict loading failed: {e2}")
                return False
        
        # Create new model instance
        new_model = FinalModelDetector(num_classes=2)
        
        # Load the state dict
        new_model.load_state_dict(state_dict)
        
        # Save just the state dict
        output_path = "final_model.pth"
        torch.save(new_model.state_dict(), output_path)
        
        print(f"✅ Successfully converted model!")
        print(f"📁 Saved state dict to: {output_path}")
        print("🎯 This file can be loaded with model.load_state_dict()")
        
        return True
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return False

if __name__ == "__main__":
    print("🔄 Model Conversion Script")
    print("Converting trained model to ML API compatible format...")
    print("=" * 50)
    
    success = convert_model()
    
    if success:
        print("=" * 50)
        print("✅ Conversion completed successfully!")
        print("💡 Now you can use the converted model in the ML API")
    else:
        print("=" * 50)
        print("❌ Conversion failed!")
        sys.exit(1)