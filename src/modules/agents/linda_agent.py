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
                                        #    nn.BatchNorm1d(args.nn_hidden_dim),
                                           activation_func,
                                           nn.Linear(args.nn_hidden_dim, args.awareness_dim * 2))
        self.mlp2 = nn.Linear(args.rnn_hidden_dim + args.n_agents * args.awareness_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.mlp1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, test_mode):
        bs = inputs.shape[0] // self.n_agents
        
        x = F.relu(self.mlp1(inputs))
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
        h = self.rnn(x, h_in) # (B * N, R)
        
        awareness_params = self.awareness_encoder(h) # (B * N, 2 * N * A)
        latent_size = self.n_agents * self.awareness_dim
        
        awareness_mean = awareness_params[:, :latent_size].reshape(bs, self.n_agents, latent_size) # (B, N, N * A)
        awareness_var = th.clamp(
            th.exp(awareness_params[:, latent_size:]), 
            min=self.var_floor).reshape(bs, self.n_agents, latent_size) # (B, N, N * A)
        
        awareness_dist = D.Normal(awareness_mean, awareness_var ** 0.5)
        awareness = awareness_dist.rsample().view(-1, latent_size)
        
        kld = th.zeros(bs, 1).to(self.args.device)
        kld0 = th.zeros(bs, 1).to(self.args.device)
        h_detach = h.view(bs, self.n_agents, self.rnn_hidden_dim).detach()
        
        if not test_mode:
            infer_input = th.zeros(self.n_agents, bs * self.n_agents, 2 * self.rnn_hidden_dim).to(self.args.device)
            for agent_i in range(self.n_agents):
                h_detach_i = h_detach[:, agent_i:agent_i+1].repeat(1, self.n_agents, 1)
                infer_input[agent_i, :, :] = th.cat([h_detach_i, h_detach], dim=-1).view(-1, 2 * self.rnn_hidden_dim)
            infer_input = infer_input.view(self.n_agents * bs * self.n_agents, 2 * self.rnn_hidden_dim) # (N * B * N, 2R)
            
            infer_params = self.infer_net(infer_input) # (N * B * N, 2A)
            infer_means = infer_params[:, :self.awareness_dim].reshape(self.n_agents, bs, latent_size) # (N * B, N * A)
            infer_vars = th.clamp(th.exp(infer_params[:, self.awareness_dim:]), 
                                    min=self.var_floor).reshape(self.n_agents, bs, latent_size) # (N, B, N * A)
            infer_means_t = th.transpose(infer_means, 0, 1) # (B, N, N * A)
            infer_vars_t = th.transpose(infer_vars, 0, 1)   # (B, N, N * A)
            
            kld = my_kld(awareness_mean, awareness_var, infer_means_t, infer_vars_t).mean(dim=-1).mean(dim=-1, keepdim=True)    
        
        # if not test_mode:
        #     for agent_i in range(self.n_agents):
        #         h_detach_i = h_detach[:, agent_i:agent_i+1].repeat(1, self.n_agents, 1)
        #         infer_input_i = th.cat([h_detach_i, h_detach], dim=-1).view(-1, 2 * self.rnn_hidden_dim) # (B * N, 2R)  # 18 hours
                
        #         infer_params = self.infer_net(infer_input_i) # (B * N, 2 * A) # 20 hours
                
        #         infer_mean = infer_params[:, :self.awareness_dim].reshape(bs, latent_size) # (B, N * A)
        #         infer_var = th.clamp(th.exp(infer_params[:, self.awareness_dim:]), 
        #                              min=self.var_floor).reshape(bs, latent_size) # (B, N * A) # 1 day 5min
                
        #         # awareness_dist_i = D.Normal(awareness_mean[:, agent_i, :], awareness_var[:, agent_i, :] ** 0.5)
        #         # infer_dist = D.Normal(infer_mean, infer_var ** 0.5)
        #         # cur_kld1 = kl_divergence(awareness_dist_i, infer_dist).sum(dim=-1).mean(dim=-1, keepdim=True)
        #         # cur_kld1 = kl_divergence(awareness_dist_i, infer_dist)
                
        #         cur_kld = my_kld(awareness_mean[:, agent_i, :], awareness_var[:, agent_i, :], infer_mean, infer_var)
        #         kld += cur_kld.mean(dim=-1, keepdim=True)
        #     kld /= self.n_agents
        # # print(kld0 - kld)
            

        h = h.view(bs * self.n_agents, -1)
        q = self.mlp2(th.cat([h, awareness], dim=-1))
        
        return q, h, kld

# def my_kld(mu1, sigma1, mu2, sigma2):
#     kld = th.log(sigma2 / sigma1) + 0.5 * (sigma1.square()+(mu1-mu2).square()) / (sigma2.square()) - 0.5
#     return kld

def my_kld(mu1, var1, mu2, var2):
    kld = 0.5 * th.log(var2 / var1) + 0.5 * (var1 + (mu1 - mu2).square()) / var2 - 0.5
    return kld

if __name__=="__main__":
    args = {
        'rnn_hidden_dim': 64,
        'nn_hidden_dim': 64,
        'n_agents': 10,
        'awareness_dim': 3,
        'var_floor': 0.002,
        'n_actions': 18,
        'device': 'cuda:0'
    }
    import argparse
    args = argparse.Namespace(**args)
    print(args)
    
    bs = 32
    inputs = th.rand(bs * args.n_agents, 204).to(args.device)
    hidden_state = th.rand(bs * args.n_agents, args.rnn_hidden_dim).to(args.device)
    
    agent = LINDAAgent(204, args).to(args.device)
    q, h, kld = agent.forward(inputs, hidden_state, test_mode=False)
    # print(q.shape, h.shape, kld.shape)
    
    from line_profiler import LineProfiler
    lp = LineProfiler()
    lp_wrapper = lp(agent.forward)
    lp_wrapper(inputs, hidden_state, False)
    lp.print_stats()