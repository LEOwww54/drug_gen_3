from collections import defaultdict

from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from autoencoder.graphtransformer import GraphAutoencoderLoss, GraphTransformerAutoencoder
import numpy as np
from autoencoder.dataprocess import from_json, SubstructureDataset, collate_graphs
from constant import device


class GraphAutoencoderTrainer:
    """训练器"""

    def __init__(self, model, train_loader, val_loader, lr=1e-4, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        self.criterion = GraphAutoencoderLoss()

    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        losses = defaultdict(float)

        for batch in tqdm(self.train_loader, desc="Training"):
            atom_feat = batch['atom_feat'].to(self.device)
            mask = batch['mask'].to(self.device)

            # 前向传播
            pred = self.model(atom_feat, mask)

            # 计算损失
            loss_dict = self.criterion(pred, batch)
            loss = loss_dict['total_loss']

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            for k, v in loss_dict.items():
                losses[k] += v.item()

        avg_loss = total_loss / len(self.train_loader)
        avg_losses = {k: v / len(self.train_loader) for k, v in losses.items()}

        return avg_loss, avg_losses

    def validate(self):
        """验证"""
        self.model.eval()
        total_loss = 0
        losses = defaultdict(float)
        reconstructions = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                atom_feat = batch['atom_feat'].to(self.device)
                mask = batch['mask'].to(self.device)

                pred = self.model(atom_feat, mask)
                loss_dict = self.criterion(pred, batch)

                total_loss += loss_dict['total_loss'].item()
                for k, v in loss_dict.items():
                    losses[k] += v.item()

                # 保存重构示例
                if len(reconstructions) < 5:
                    z = pred['z']
                    smiles = self.model.decode_to_graph(z[0])
                    if smiles:
                        reconstructions.append({
                            'input': batch['smiles'][0],
                            'output': smiles
                        })

        avg_loss = total_loss / len(self.val_loader)
        avg_losses = {k: v / len(self.val_loader) for k, v in losses.items()}

        return avg_loss, avg_losses, reconstructions

    def train(self, epochs=100, save_path='best_model.pth'):
        """完整训练"""
        best_val_loss = float('inf')

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            # 训练
            train_loss, train_losses = self.train_epoch()
            print(f"Train Loss: {train_loss:.4f}")
            print(f"  Atom: {train_losses['atom_loss']:.4f}, Edge: {train_losses['edge_loss']:.4f}")

            # 验证
            val_loss, val_losses, reconstructions = self.validate()
            print(f"Val Loss: {val_loss:.4f}")

            # 打印重构示例
            if reconstructions:
                print("\nReconstruction examples:")
                for rec in reconstructions[:3]:
                    print(f"  Input: {rec['input']}")
                    print(f"  Output: {rec['output']}")

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                }, save_path)
                print(f"✓ Model saved to {save_path}")

            self.scheduler.step()

def train(smiles_list):
    dataset = SubstructureDataset(smiles_list, max_nodes=70)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        collate_fn=collate_graphs,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_graphs,
        num_workers=0
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"Atom vocab size: {dataset.atom_vocab_size}")
    print(f"Bond vocab size: {dataset.bond_vocab_size}")

    # 3. 创建自编码器
    max_nodes = 70  # 提取为变量
    autoencoder = GraphTransformerAutoencoder(
        atom_vocab_size=dataset.atom_vocab_size,
        bond_vocab_size=dataset.bond_vocab_size,
        d_model=64,
        latent_dim=128,
        n_heads=8,
        n_layers=8,
        max_nodes=max_nodes,
        dropout=0.1
    )

    print(f"\nModel parameters: {sum(p.numel() for p in autoencoder.parameters()):,}")

    # 4. 训练
    trainer = GraphAutoencoderTrainer(
        model=autoencoder,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=1e-4,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    trainer.train(epochs=50, save_path='graph_autoencoder.pth')

    # 5. 生成新分子
    print("\nGenerating new molecules...")
    generated = []
    for i in range(10):
        z = torch.randn(1, autoencoder.latent_dim)
        smiles = autoencoder.decode_to_graph(z, max_nodes=max_nodes)  # 添加max_nodes参数
        if smiles:
            generated.append(smiles)

    print("\nGenerated molecules:")
    for i, smiles in enumerate(generated):
        print(f"  {i + 1}. {smiles}")

if __name__ == "__main__":
    smiles_list = from_json('../data/stru_data.json', 0)[1]
    train(smiles_list)