"""
Transformer-based classifier for MassLinker metabolic tokens.

This module defines a Transformer encoder architecture for supervised
classification using MassLinker tokenized LC-MS metabolomics data. Each sample
is represented as a sequence of compound-level metabolic tokens, where each
token contains RBF-derived features describing chromatographic peak height,
position, and width.

The model projects each token into an embedding space, applies stacked
Transformer encoder blocks to capture high-dimensional dependencies among
metabolic tokens, and uses fully connected layers for final classification.
A weighted focal loss is provided to address class imbalance during training.
"""

from data import ExcelDataset
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import logging
from datetime import datetime


class transformer_encoder(nn.Module):
    """
    Stacked Transformer encoder for MassLinker token embeddings.

    The encoder consists of multiple one-layer TransformerEncoder modules.
    Each block applies multi-head self-attention and a feed-forward network to
    model dependencies across compound-level MassLinker tokens.

    Parameters
    ----------
    embed_dim : int, optional
        Token embedding dimension.
    n_heads : int, optional
        Number of self-attention heads.
    dim_feedforward : int, optional
        Dimension of the feed-forward network inside each Transformer block.
    dropout : float, optional
        Dropout rate used in Transformer encoder layers.
    num_layers : int, optional
        Number of Transformer encoder blocks.
    """

    def __init__(self, embed_dim=512, n_heads=16, dim_feedforward=2048, dropout=0.25, num_layers=12):
        """
        Initialize stacked Transformer encoder blocks.
        """
        super().__init__()

        self.encoder = nn.ModuleList([nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads,
                                                                                       activation=nn.GELU(),
                                                                                       dim_feedforward=dim_feedforward,
                                                                                       dropout=dropout,
                                                                                       batch_first=True,
                                                                                       ),
                                                            num_layers=1,
                                                            norm=nn.LayerNorm(embed_dim)) for _ in range(num_layers)])

    def forward(self, x):
        """
        Forward pass through the stacked Transformer encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input token embeddings with shape approximately
            batch_size x sequence_length x embed_dim.

        Returns
        -------
        torch.Tensor
            Contextualized token embeddings.
        """
        for encoder in self.encoder:
            x = encoder(x)
        return x


class transformer_language(nn.Module):
    """
    Transformer classifier for MassLinker metabolic token sequences.

    Each MassLinker sample is treated as a sequence of metabolic tokens. The
    token features are first projected into an embedding space, processed by
    Transformer encoder blocks, flattened, and then passed through fully
    connected layers for classification.

    Parameters
    ----------
    in_chans : int, optional
        Number of input features per token. For MassLinker, the default is 60,
        corresponding to 3 RBF parameter types and 20 RBF components.
    embed_dim : int, optional
        Token embedding dimension.
    depth : int, optional
        Number of Transformer encoder blocks.
    n_heads : int, optional
        Number of self-attention heads.
    seq_len : int, optional
        Number of MassLinker metabolic tokens in each sample.
    class_count : int, optional
        Number of output classes.
    """

    def __init__(self, in_chans=60, embed_dim=512, depth=12, n_heads=16,
                 seq_len=5194, class_count=2):
        """
        Initialize the Transformer-based classification model.
        """
        super().__init__()

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.depth = depth
        self.n_heads = n_heads

        # Project RBF-derived token features into the Transformer embedding space.
        self.linear = nn.Linear(in_chans, embed_dim)

        self.transformer = transformer_encoder(embed_dim=embed_dim, n_heads=n_heads, num_layers=depth)
        self.flattener = nn.Flatten()

        # Classification head.
        self.fc0 = nn.Linear(embed_dim * seq_len, int(embed_dim / 2))
        self.fc1 = nn.Linear(int(embed_dim / 2), class_count)

        self.layer_norm = nn.LayerNorm(embed_dim)

        # Apply truncated normal initialization to supported layers.
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Initialize model weights using truncated normal initialization.

        Parameters
        ----------
        m : torch.nn.Module
            Module to initialize.
        """
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass of the Transformer classifier.

        Parameters
        ----------
        x : torch.Tensor
            Input MassLinker token tensor with shape approximately
            batch_size x seq_len x in_chans.

        Returns
        -------
        torch.Tensor
            Raw class logits.
        """
        # Token-wise linear projection followed by layer normalization and GELU.
        x = F.gelu(self.layer_norm(self.linear(x)))

        # Contextualize MassLinker tokens using Transformer self-attention.
        x = self.transformer(x)

        # Flatten token embeddings and apply classification layers.
        x = F.gelu(self.flattener(x))
        x = F.gelu(self.fc0(x))
        x = self.fc1(x)

        # x = F.softmax(x, dim=1)
        return x


class WeightedFocalLoss(nn.Module):
    """
    Weighted focal loss for class-imbalanced MassLinker classification.

    The focal loss down-weights easy samples and emphasizes hard-to-classify
    samples. Class-specific weights can be provided through `alpha`, which is
    useful when disease/control or multiclass sample distributions are imbalanced.

    Parameters
    ----------
    alpha : float or torch.Tensor, optional
        Global or class-specific weighting factor.
    gamma : float, optional
        Focusing parameter that controls the strength of hard-sample emphasis.
    reduction : str, optional
        Reduction mode. Supported values are `mean`, `sum`, or no reduction.
    """

    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        """
        Initialize weighted focal loss.
        """
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Compute weighted focal loss.

        Parameters
        ----------
        inputs : torch.Tensor
            Raw model logits with shape batch_size x class_count.
        targets : torch.Tensor
            Ground-truth class labels.

        Returns
        -------
        torch.Tensor
            Focal loss value.
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)

        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def get_class_weights(y_train):
    """
    Calculate balanced class weights from training labels.

    The weights are computed using inverse-frequency balancing and can be used
    as class-specific alpha values in the weighted focal loss.

    Parameters
    ----------
    y_train : torch.Tensor
        Training labels.

    Returns
    -------
    torch.FloatTensor
        Class weights.
    """
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train.numpy())
    return torch.FloatTensor(weights)


if __name__ == "__main__":
    """
    Train the Transformer classifier on MassLinker datasets.

    This training script loads a merged MassLinker dataset, splits original
    samples into training and test partitions, expands the training partition
    with augmented samples, and trains the Transformer classifier using weighted
    focal loss.

    The best model is saved according to test accuracy, and the final model is
    saved after all epochs.
    """
    log_filename = f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load the full MassLinker dataset containing original samples.
    dataset = joblib.load(r"merged_dataset_enhanced_total.joblib")

    # Initialize the Transformer classifier.
    model = transformer_language(class_count=len(set(dataset.classes)), embed_dim=512, n_heads=8,
                                 depth=8).to(device)

    # Convert MassLinker token tensors into sequence format:
    # sample_count x token_count x 60 RBF-derived features.
    x = dataset.samples
    x = torch.reshape(x, (x.shape[0], x.shape[1], 60))

    y = dataset.is_positive
    y = torch.tensor(y, dtype=torch.long)

    # x_train, x_test, y_train, y_test = train_test_split(
    #     x, y, test_size=0.2, random_state=42, stratify=y
    # )

    # Split original samples into training and test partitions.
    train_idx, test_idx = train_test_split(
        [i for i in range(len(x))], test_size=0.2, random_state=42, stratify=y
    )

    x_test = x[[i for i in test_idx]]
    y_test = torch.tensor(y[[i for i in test_idx]], dtype=torch.long)

    # Load the augmented MassLinker dataset and use augmented samples only for
    # training, avoiding augmentation leakage into the test set.
    enhanced_dataset = joblib.load(r"merged_dataset_enhanced.joblib")

    train_enhanced_idx = []
    train_x_name = [dataset.name[i] for i in train_idx]

    for i in range(len(enhanced_dataset)):
        if enhanced_dataset.name[i] in train_x_name:
            train_enhanced_idx.append(i)

    x_enhanced = enhanced_dataset.samples
    x_enhanced = torch.reshape(x_enhanced, (x_enhanced.shape[0], x_enhanced.shape[1], 60))

    y_enhanced = enhanced_dataset.is_positive
    y_enhanced = torch.tensor(y_enhanced, dtype=torch.long)

    x_train = x_enhanced[[i for i in train_enhanced_idx]]
    y_train = y_enhanced[[i for i in train_enhanced_idx]]

    #joblib.dump([train_idx, test_idx], filename="train_test_index.joblib")

    # Compute class weights for weighted focal loss.
    class_weights = get_class_weights(y_train).to(device)
    criterion = WeightedFocalLoss(alpha=class_weights, gamma=2)

    num_epochs = 200
    batch_size = 16
    best_accuracy = 0

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    for epoch in tqdm(range(num_epochs)):
        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()

        model.eval()

        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                test_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                test_total += batch_y.size(0)
                test_correct += (predicted == batch_y).sum().item()

        train_accuracy = 100 * train_correct / train_total
        test_accuracy = 100 * test_correct / test_total

        scheduler.step()

        logger.info(f'Epoch [{epoch + 1}/{num_epochs}]')
        logger.info(f'Train loss: {train_loss / len(train_loader):.4f}, Train accuracy: {train_accuracy:.2f}%')
        logger.info(f'Test loss: {test_loss / len(test_loader):.4f}, Test accuracy: {test_accuracy:.2f}%')
        logger.info(f'Learning rate: {optimizer.param_groups[0]["lr"]:.6f}')
        logger.info('-' * 60)

        # print(f'Epoch [{epoch + 1}/{num_epochs}]')
        # print(f'Train loss: {train_loss / len(train_loader):.4f}, Train accuracy: {train_accuracy:.2f}%')
        # print(f'Test loss: {test_loss / len(test_loader):.4f}, Test accuracy: {test_accuracy:.2f}%')
        # print(f'Learning rate: {optimizer.param_groups[0]["lr"]:.6f}')
        # print('-' * 60)

        if test_accuracy >= best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            logger.info(f'Best model saved. Test accuracy: {best_accuracy:.2f}%')
            # print(f'Best model saved. Test accuracy: {best_accuracy:.2f}%')

    # print(f"Training completed. Best test accuracy: {best_accuracy:.2f}%")
    logger.info(f"Training completed. Best test accuracy: {best_accuracy:.2f}%")

    torch.save(model.state_dict(), 'final_model.pth')
