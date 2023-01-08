import torch as th
import torch.nn as nn
import torch.nn.functional as F

import torch.distributions as D
from torch.distributions.kl import kl_divergence


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = th.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = th.matmul(attn, v)

        return output, attn


# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/fec78a687210851f055f792d45300d27cc60ae41/transformer/SubLayers.py#L9
class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        '''
        (B, N, d_model) -> q: (B, N, n_head * d_k) 
                           k: (B, N, n_head * d_k)
                           v: (B, N, n_head * d_v)
                        -> attn: (B, n_head, N, N)
                           v: (B, n_head, N, d_v)
                        -> out: (B, n_head, N, d_v)
                        -> reshape: (B, N, n_head * d_v)
                        -> fc: (B, N, d_model)
                        -> dropout, residual, layer_norm
        '''
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn


class LINDAAttenAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(LINDAAttenAgent, self).__init__()
        
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
        
        self.multi_head_attention = MultiHeadAttention(n_head=4, d_model=args.awareness_dim, d_k=32, d_v=32)

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

        h = h.view(bs * self.n_agents, -1)
        atten_in = awareness.view(bs * self.args.n_agents, self.args.n_agents, -1) # (B * N, N, A)
        atten_out, atten = self.multi_head_attention(atten_in, atten_in, atten_in) # atten_out: (B * N, N, A), atten: (B * N, n_head, N, N)
        atten_out = atten_out.view(bs * self.n_agents, -1)
        q = self.mlp2(th.cat([h, atten_out], dim=-1))
        
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
    
    # from line_profiler import LineProfiler
    # lp = LineProfiler()
    # lp_wrapper = lp(agent.forward)
    # lp_wrapper(inputs, hidden_state, False)
    # lp.print_stats()