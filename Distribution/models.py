# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence
)
import numpy as np
import matplotlib.pyplot as plt

class trajectory2seq(nn.Module):
    def __init__(self, hidden_dim, n_layers, int2symb, symb2int, dict_size, device, maxlen):
        super(trajectory2seq, self).__init__()
        # Definition des parametres
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device
        self.symb2int = symb2int
        self.int2symb = int2symb
        self.dict_size = dict_size
        self.maxlen = maxlen

        # Definition des couches
        # Couches pour rnn

        # Encodeur bidirectionnel
        self.encoder = nn.GRU(
            input_size=2, # x, y
            hidden_size=self.hidden_dim,
            num_layers=self.n_layers,
            batch_first=True,
            bidirectional=True
        )

        # Décodeur unidirectionnel
        self.decoder = nn.GRU(
            input_size=self.dict_size,
            hidden_size=2 * self.hidden_dim,
            num_layers=self.n_layers,
            batch_first=True,
            bidirectional=False
        )

        # Couches pour attention
        # À compléter

        # Couche dense pour la sortie
        self.fc = nn.Linear(
            in_features=4 * self.hidden_dim,
            out_features=self.dict_size
        )

    def forward(self, x, lengths):
        # =============================================
        # 1. Préparation des entrées
        # =============================================

        # (N, 2, T) -> (N, T, 2)
        x = x.transpose(1, 2)

        N, T, _ = x.shape

        lengths = torch.as_tensor(
            lengths, dtype=torch.long, device="cpu"
        )

        # =============================================
        # 2. Encodeur bidirectionnel
        # =============================================

        packed = pack_padded_sequence(
            x,
            lengths,
            batch_first=True,
            enforce_sorted=False
        )

        packed_values, h_enc = self.encoder(packed)

        values, _ = pad_packed_sequence(
            packed_values,
            batch_first=True,
            total_length=T
        )

        # Concaténation des contextes finaux
        # (2, N, 16) -> (1, N, 32)
        h_dec = torch.cat(
            (h_enc[0], h_enc[1]), dim=1
        ).unsqueeze(0)

        # Masque des points valides
        mask = (
                torch.arange(T, device=x.device)[None, :]
                < lengths.to(x.device)[:, None]
        )

        # =============================================
        # 3. Initialisation du décodeur
        # =============================================

        decoder_input = x.new_zeros(
            N, 1, self.dict_size
        )

        decoder_input[:, :, self.symb2int["<sos>"]] = 1

        predictions = []
        attention_weights = []

        # =============================================
        # 4. Décodage
        # =============================================

        for j in range(self.maxlen):
            # Contexte du décodeur
            q, h_dec = self.decoder(
                decoder_input, h_dec
            )

            # -----------------------------------------
            # 5. Attention de Luong
            # -----------------------------------------

            # Produit scalaire
            scores = torch.bmm(
                q, values.transpose(1, 2)
            )

            # Ignorer les points de remplissage
            scores = scores.masked_fill(
                ~mask.unsqueeze(1),
                float("-inf")
            )

            # Poids d'attention
            w = torch.softmax(scores, dim=-1)

            # Vecteur d'attention
            a = torch.bmm(w, values)

            # -----------------------------------------
            # 6. Classification
            # -----------------------------------------

            combined = torch.cat((q, a), dim=-1)

            logits = self.fc(combined)

            predictions.append(logits.squeeze(1))

            attention_weights.append(
                w.squeeze(1).detach()
            )

            # -----------------------------------------
            # 7. Entrée de la prochaine itération
            # -----------------------------------------

            token = logits.squeeze(1).argmax(dim=1)

            decoder_input = F.one_hot(
                token,
                num_classes=self.dict_size
            ).to(x.dtype).unsqueeze(1)

        # =============================================
        # 8. Préparation des sorties
        # =============================================

        self.attention_weights = torch.stack(
            attention_weights, dim=1
        )

        return torch.stack(predictions, dim=2)

    


