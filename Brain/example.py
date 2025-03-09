import os
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from metrics import create_metrics, DualTaskLoss
from src.model import MobileNetUNet
from dataset import get_data_loaders
from trainer import ModelTrainer
from pathlib import Path

def main():
    # Configuration
    CONFIG = {
        'data_dir': r"D:\FPT BT\DBM\final project\BrainTumor_Split_mask_aug",
        'batch_size': 16,
        'num_workers': 5,
        'learning_rate': 0.0025,
        'num_epochs': 150,
        'patience': 100,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seg_weight': 1.2,
        'cls_weight': 0.4,                              
        'save_dir': 'checkpoint_new'  
    }

    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    save_path = os.path.join(CONFIG['save_dir'], 'best_model.pt')

    # 1. Initialize model
    model = MobileNetUNet(                         
        img_ch=1,
        seg_ch=4,
        num_classes=3
    ).to(CONFIG['device'])

    # 2. Create data loaders
    train_loader, val_loader = get_data_loaders(
        root_dir=CONFIG['data_dir'],
        batch_size=CONFIG['batch_size'],
        num_workers=CONFIG['num_workers']
    )

    # Check the number of samples in the data loaders
    print(f"Number of training samples: {len(train_loader.dataset)}")
    print(f"Number of validation samples: {len(val_loader.dataset)}")

    # 3. Setup metrics
    metrics = create_metrics()

    # 4. Initialize optimizer and loss functions
    optimizer = Adam(
        model.parameters(), 
        lr=CONFIG['learning_rate'],
        weight_decay=0.0001,
        betas=(0.9, 0.999)
    )
    
    criterion = DualTaskLoss(
        seg_weight=CONFIG['seg_weight'],
        cls_weight=CONFIG['cls_weight']
    )
    
    # 5. Initialize scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        patience=10,
        factor=0.5,
        verbose=True,
        min_lr=0.0001
    )

    # 6. Initialize trainer
    trainer = ModelTrainer(
        model=model,
        dataloaders={
            'train': train_loader,
            'val': val_loader
        },
        criterion_seg=criterion.seg_criterion,
        criterion_cls=criterion.cls_criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        metrics=metrics,
        device=CONFIG['device'],
        patience=CONFIG['patience'],
        task_weights={
            'seg': CONFIG['seg_weight'],
            'cls': CONFIG['cls_weight']
        }
    )

    # 7. Train the model
    model = trainer.train(
        num_epochs=CONFIG['num_epochs'],
        save_path=save_path
    )
    
    # Save final model
    final_save_path = os.path.join(CONFIG['save_dir'], 'final_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'config': CONFIG
    }, final_save_path)

    print("Training completed successfully!")

if __name__ == "__main__":
    main()