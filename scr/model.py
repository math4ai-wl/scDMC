"""
Variational autoencoder model for scRNA-seq (ZINB reconstruction).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ZINBLoss(nn.Module):
    def __init__(self):
        super(ZINBLoss, self).__init__()

    def forward(self, x, mean, disp, pi, scale_factor=1.0, ridge_lambda=0.0):
        eps = 1e-10
        scale_factor = scale_factor[:, None]
        mean = mean * scale_factor

        t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
        t2 = (disp + x) * torch.log(1.0 + (mean / (disp + eps))) + (
            x * (torch.log(disp + eps) - torch.log(mean + eps))
        )
        nb_final = t1 + t2

        nb_case = nb_final - torch.log(1.0 - pi + eps)
        zero_nb = torch.pow(disp / (disp + mean + eps), disp)
        zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)
        result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)

        if ridge_lambda > 0:
            ridge = ridge_lambda * torch.square(pi)
            result += ridge

        result = torch.mean(result)
        return result


class MeanAct(nn.Module):
    def __init__(self):
        super(MeanAct, self).__init__()

    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)


class DispAct(nn.Module):
    def __init__(self):
        super(DispAct, self).__init__()

    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)


class VariationalAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim,
        z_dim,
        encode_layers=None,
        decode_layers=None,
        activation="relu",
        sigma=1,
        alpha=1.0,
        gamma=1.0,
        device="cuda",
    ):
        super(VariationalAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.encode_layers = encode_layers or []
        self.decode_layers = decode_layers or []
        self.gamma = gamma
        self.alpha = alpha
        self.sigma = sigma
        self.device = device
        self.activation = activation

        # Encoder output dimension matches the last hidden size for mu/logvar heads.
        if self.encode_layers:
            enc_out_dim = self.encode_layers[-1]
            enc_hidden = self.encode_layers[:-1]
        else:
            enc_out_dim = self.z_dim
            enc_hidden = []
        self.encoder = self.build_network(self.input_dim, enc_out_dim, enc_hidden, self.activation)

        # Decoder maps latent z back to input dimension via ZINB parameter heads.
        if self.decode_layers:
            dec_out_dim = self.decode_layers[-1]
            dec_hidden = self.decode_layers[:-1]
        else:
            dec_out_dim = self.input_dim
            dec_hidden = []
        self.decoder = self.build_network(self.z_dim, dec_out_dim, dec_hidden, self.activation)

        self.enc_mu = nn.Linear(enc_out_dim, self.z_dim)
        self.enc_logvar = nn.Linear(enc_out_dim, self.z_dim)

        self.dec_mean = nn.Sequential(nn.Linear(dec_out_dim, self.input_dim), MeanAct())
        self.dec_disp = nn.Sequential(nn.Linear(dec_out_dim, self.input_dim), DispAct())
        self.dec_pi = nn.Sequential(nn.Linear(dec_out_dim, self.input_dim), nn.Sigmoid())

        self.zinb_loss = ZINBLoss()
        self.to(self.device)

    def encode(self, x):
        """
        Encode inputs into the latent mean representation (e.g., for clustering).
        """
        h = self.encoder(x)
        mu = self.enc_mu(h)
        return mu

    def encode_vae(self, x):
        h = self.encoder(x)
        mu = self.enc_mu(h)
        logvar = self.enc_logvar(h)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar

    def build_network(self, input_dim, output_dim, hidden_layers=None, activation="relu"):
        hidden_layers = hidden_layers or []
        layers = [input_dim] + hidden_layers + [output_dim]
        network = []
        for i in range(1, len(layers)):
            network.append(nn.Linear(layers[i - 1], layers[i]))
            if i < len(layers) - 1:
                network.append(nn.ReLU() if activation == "relu" else nn.Tanh())
        return nn.Sequential(*network)

    def forward_AE(self, x):
        """
        Pretraining forward pass: reconstruction only (no clustering objective).
        """
        h = self.encoder(x + torch.randn_like(x) * self.sigma)
        z = self.enc_mu(h)
        h = self.decoder(z)
        mean = self.dec_mean(h)
        disp = self.dec_disp(h)
        pi = self.dec_pi(h)

        h0 = self.encoder(x)
        z0 = self.enc_mu(h0)
        return z0, mean, disp, pi
