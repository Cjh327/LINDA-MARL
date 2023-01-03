import torch as th
import torch.nn as nn
import torch.nn.functional as F

import torch.distributions as D
from torch.distributions.kl import kl_divergence


class LINDAAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(LINDAAgent, self).__init__()
        self.args = args
        
        self.rnn_hidden_dim = args.rnn_hidden_dim
        self.nn_hidden_dim = args.nn_hidden_dim
        self.n_agents = args.n_agents
        self.awareness_dim = args.awareness_dim
        self.var_floor = args.var_floor

        activation_func = nn.LeakyReLU()

        self.mlp1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.awareness_encoder = nn.Sequential(nn.Linear(args.rnn_hidden_dim, args.nn_hidden_dim),
                                               nn.BatchNorm1d(args.nn_hidden_dim),
                                               activation_func,
                                               nn.Linear(args.nn_hidden_dim, args.n_agents * args.awareness_dim * 2))
        self.infer_net = nn.Sequential(nn.Linear(2 * args.rnn_hidden_dim, args.nn_hidden_dim),
                                           nn.BatchNorm1d(args.nn_hidden_dim),
                                           activation_func,
                                           nn.Linear(args.nn_hidden_dim, args.awareness_dim * 2))
        self.mlp2 = nn.Linear(args.rnn_hidden_dim + args.n_agents * args.awareness_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.mlp1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, test_mode=False):
        # print('inputs shape', inputs.shape)
        bs = inputs.shape[0] // self.n_agents
        
        x = F.relu(self.mlp1(inputs))
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
        h = self.rnn(x, h_in) # (B * N, R)
        
        # print('h shape', h.shape)
        
        awareness_params = self.awareness_encoder(h) # (B * N, 2 * N * R)
        latent_size = self.n_agents * self.awareness_dim
        
        awareness_mean = awareness_params[:, :latent_size].reshape(bs, self.n_agents, latent_size) # (B, N, N * R)
        awareness_var = th.clamp(
            th.exp(awareness_params[:, latent_size:]), 
            min=self.var_floor).reshape(bs, self.n_agents, latent_size) # (B, N, N * R)
        
        awareness_dist = D.Normal(awareness_mean, awareness_var ** 0.5)
        awareness = awareness_dist.rsample().view(-1, latent_size)
        
        # print('awareness shape', awareness_mean.shape, awareness_var.shape)
        
        kld = th.zeros(bs, 1).to(self.args.device)
        h_detach = h.view(bs, self.n_agents, self.rnn_hidden_dim).detach()
        if not test_mode:
            for agent_i in range(self.n_agents):
                h_detach_i = h_detach[:, agent_i:agent_i+1].repeat(1, self.n_agents, 1)
                infer_input_i = th.cat([h_detach_i, h_detach], dim=-1).view(-1, 2 * self.rnn_hidden_dim)
                infer_params = self.infer_net(infer_input_i) # (B * N, 2 * R)
                infer_mean = infer_params[:, :self.awareness_dim].reshape(bs, latent_size) # (B, N * R)
                infer_var = th.clamp(th.exp(infer_params[:, self.awareness_dim:]), 
                                     min=self.var_floor).reshape(bs, latent_size) # (B, N * R)
                
                awareness_dist_i = D.Normal(awareness_mean[:, agent_i, :], awareness_var[:, agent_i, :])
                infer_dist = D.Normal(infer_mean, infer_var ** 0.5)
                # print('kld', kl_divergence(awareness_dist, infer_dist).shape)
                # print('kld', awareness_dist_i.rsample().shape, infer_dist.rsample().shape)
                
                kld += kl_divergence(awareness_dist_i, infer_dist).sum(dim=-1).mean(dim=-1, keepdim=True)
                # print(kl_divergence(awareness_dist_i, infer_dist).sum(dim=-1).mean(dim=-1).shape, kld.shape)
                

        h = h.view(bs * self.n_agents, -1)
        q = self.mlp2(th.cat([h, awareness], dim=-1))
        
        return q, h, kld
