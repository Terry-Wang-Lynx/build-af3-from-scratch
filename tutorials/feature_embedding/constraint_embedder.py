# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention.linear import LinearNoBias
from feature_embedding.atom_attention import AtomAttentionEncoder
from runtime.logger import get_logger

logger = get_logger(__name__)



class SubstructureEmbedder(nn.Module):
    """
    Implements Substructure Embedder

    Args:
        n_classes (int): Number of distance classes in input
        c_pair_dim (int): Output pair embedding dimension
        architecture (str): Either 'mlp' or 'transformer'
        hidden_dim (int): Hidden dimension for both architectures
        n_layers (int): Number of layers (MLP or Transformer)
        dropout (float): Dropout rate
    """

    def __init__(
        self,
        n_classes: int,
        c_pair_dim: int,
        architecture: str = "mlp",
        hidden_dim: int = 256,
        n_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.architecture = architecture.lower()
        if self.architecture == "mlp":
            layers = []
            layers.append(LinearNoBias(n_classes, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            for _ in range(n_layers - 2):
                layers.append(LinearNoBias(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))

            # Output layer
            layers.append(LinearNoBias(hidden_dim, c_pair_dim))

            self.network = nn.Sequential(*layers)

        elif self.architecture == "transformer":
            self.input_proj = LinearNoBias(n_classes, hidden_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.output_proj = LinearNoBias(hidden_dim, c_pair_dim)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input distance map
                shape: [..., N_token, N_token, N_classes]
        Returns:
            torch.Tensor: Output pair embeddings
                shape: [..., N_token, N_token, c_pair_dim]
        """
        if self.architecture == "mlp":
            return self.network(x)
        else:  # transformer
            # Reshape for transformer
            orig_shape = x.shape
            x = x.view(-1, orig_shape[-3], orig_shape[-2], orig_shape[-1])
            batch_size, n_token, _, n_classes = x.shape

            # Project and reshape to sequence
            x = self.input_proj(x)
            x = x.reshape(batch_size, n_token * n_token, -1)

            # Apply transformer
            x = self.transformer(x)

            # Project to output dim and reshape back
            x = self.output_proj(x)
            x = x.reshape(*orig_shape[:-1], -1)

            return x



class ConstraintEmbedder(nn.Module):
    """
    Implements Constraint Embedder

    Args:
        pocket_embedder (dict[str, Any]): pocket embedder config
        contact_embedder (dict[str, Any]): contact embedder config
        contact_atom_embedder (dict[str, Any]): contact atom embedder config
        substructure_embedder (dict[str, Any]): substructure embedder config
        c_constraint_z (int): constraint z dimension
        initialize_method (str): initialize method
    """

    def __init__(
        self,
        pocket_embedder: dict[str, Any],
        contact_embedder: dict[str, Any],
        contact_atom_embedder: dict[str, Any],
        substructure_embedder: dict[str, Any],
        c_constraint_z: int,
        initialize_method: str = "zero",
        **kwargs: Any,
    ) -> None:
        super(ConstraintEmbedder, self).__init__()
        self.pocket_embedder_config = pocket_embedder
        self.contact_embedder_config = contact_embedder
        self.contact_atom_embedder_config = contact_atom_embedder
        self.substructure_embedder_config = substructure_embedder
        # pocket embedder
        if self.pocket_embedder_config.get("enable", False):
            self.pocket_z_embedder = LinearNoBias(
                in_features=self.pocket_embedder_config.get("c_z_input", 1),
                out_features=c_constraint_z,
            )

        # token contact embedder
        if self.contact_embedder_config.get("enable", False):
            self.contact_z_embedder = LinearNoBias(
                in_features=contact_embedder["c_z_input"], out_features=c_constraint_z
            )

        # atom contact embedder
        if self.contact_atom_embedder_config.get("enable", False):
            self.contact_atom_z_embedder = LinearNoBias(
                in_features=contact_atom_embedder["c_z_input"],
                out_features=c_constraint_z,
            )

        # substructure embedder
        if self.substructure_embedder_config.get("enable", False):
            self.substructure_z_embedder = SubstructureEmbedder(
                n_classes=self.substructure_embedder_config.get("n_classes", 4),
                c_pair_dim=c_constraint_z,
                architecture=self.substructure_embedder_config.get(
                    "architecture", "mlp"
                ),
                hidden_dim=self.substructure_embedder_config.get("hidden_dim", 256),
                n_layers=self.substructure_embedder_config.get("n_layers", 3),
            )

        for module in self.modules():
            if isinstance(module, nn.Linear):
                if initialize_method == "zero":
                    nn.init.zeros_(module.weight)

    def forward(
        self,
        constraint_feature_dict: dict[str, Any],
    ) -> torch.Tensor:
        """

        Args:
            constraint_feature_dict (dict[str, Any]): dict of input features

        Returns:
            torch.Tensor: token embedding
                [..., N_token, c_s]
        """
        z_constraint = None

        if self.pocket_embedder_config.get("enable", False):
            z_constraint = self.pocket_z_embedder(constraint_feature_dict["pocket"])

        if self.contact_embedder_config.get("enable", False):
            z_contact = self.contact_z_embedder(constraint_feature_dict["contact"])
            z_constraint = (
                z_contact if z_constraint is None else z_constraint + z_contact
            )

        if self.contact_atom_embedder_config.get("enable", False):
            z_contact_atom = self.contact_atom_z_embedder(
                constraint_feature_dict["contact_atom"]
            )
            z_constraint = (
                z_contact_atom
                if z_constraint is None
                else z_constraint + z_contact_atom
            )

        # substructure embedder
        if self.substructure_embedder_config.get("enable", False):
            z_substructure = self.substructure_z_embedder(
                constraint_feature_dict["substructure"]
            )
            z_constraint = (
                z_substructure
                if z_constraint is None
                else z_constraint + z_substructure
            )
        return z_constraint
