from collections import defaultdict

from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from autoencoder.graphtransformer import GraphAutoencoderLoss, GraphTransformerAutoencoder
import numpy as np
from autoencoder.dataprocess import from_json, SubstructureDataset, collate_graphs


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
            node_feat = batch['node_feat'].to(self.device)
            edge_feat = batch['edge_feat'].to(self.device)
            mask = batch['mask'].to(self.device)

            # 前向传播
            pred = self.model(node_feat, edge_feat, mask)

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
                node_feat = batch['node_feat'].to(self.device)
                edge_feat = batch['edge_feat'].to(self.device)
                mask = batch['mask'].to(self.device)

                # 前向传播
                pred = self.model(node_feat, edge_feat, mask)

                # 计算损失
                loss_dict = self.criterion(pred, batch)
                loss = loss_dict['total_loss']

                total_loss += loss.item()
                for k, v in loss_dict.items():
                    losses[k] += v.item()

                # 保存重构示例
                if len(reconstructions) < 10:
                    z = pred['z']
                    rec_smiles = self.model.decode_to_smiles(z[0])
                    if rec_smiles:
                        reconstructions.append({
                            'input': batch['smiles'][0],
                            'output': rec_smiles
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
            print(
                f"  Atom: {train_losses['atom_loss']:.4f}, Edge: {train_losses['edge_exist_loss']:.4f}, Length: {train_losses['length_loss']:.4f}")

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

            # 学习率调度
            self.scheduler.step()

    def generate_molecules(self, n_samples=10, length_range=(5, 20)):
        """生成新分子"""
        self.model.eval()
        generated = []

        with torch.no_grad():
            for _ in range(n_samples):
                # 从潜在空间采样
                z = torch.randn(1, self.model.latent_dim).to(self.device)

                # 设置随机长度
                length = torch.randint(length_range[0], length_range[1] + 1, (1,)).to(self.device)

                # 解码
                smiles = self.model.decode_to_smiles(z)
                if smiles:
                    generated.append(smiles)

        return generated

    def interpolate(self, smiles1, smiles2, n_steps=10):
        """在潜在空间插值"""
        self.model.eval()

        z1 = self.model.encode_smiles(smiles1).to(self.device)
        z2 = self.model.encode_smiles(smiles2).to(self.device)

        interpolated = []
        for alpha in np.linspace(0, 1, n_steps):
            z = (1 - alpha) * z1 + alpha * z2
            smiles = self.model.decode_to_smiles(z)
            if smiles:
                interpolated.append({
                    'alpha': alpha,
                    'smiles': smiles
                })

        return interpolated

def train(smiles_list):
    dataset = SubstructureDataset(smiles_list, max_nodes=30)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_graphs)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_graphs)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"Node features: {dataset[0]['node_feat'].shape[-1]}")
    print(f"Edge features: {dataset[0]['edge_feat'].shape[-1]}")

    # 3. 创建模型
    model = GraphTransformerAutoencoder(
        node_dim=6,
        edge_dim=5,
        d_model=256,
        latent_dim=128,
        n_heads=8,
        n_layers=4,
        max_nodes=30,
        dropout=0.1
    )

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 4. 训练
    trainer = GraphAutoencoderTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=1e-4,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    trainer.train(epochs=50, save_path='graph_autoencoder.pth')

    # 5. 生成新分子
    print("\nGenerating new molecules...")
    generated = trainer.generate_molecules(n_samples=10, length_range=(5, 15))
    print("\nGenerated molecules:")
    for i, smiles in enumerate(generated):
        print(f"  {i + 1}. {smiles}")