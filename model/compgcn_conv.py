from helper import *
from model.message_passing import MessagePassing

class CompGCNConv(MessagePassing):
	def __init__(self, in_channels, out_channels, num_rels, act=lambda x:x, params=None):
		super(self.__class__, self).__init__()

		self.p 			= params
		self.in_channels	= in_channels
		self.out_channels	= out_channels
		self.num_rels 		= num_rels
		self.act 		= act
		self.device		= None
		self.leakyrelu = torch.nn.LeakyReLU(0.2)

		self.w_loop		= get_param((in_channels, out_channels))
		self.w_in		= get_param((in_channels, out_channels))
		self.w_out		= get_param((in_channels, out_channels))
		self.w_rel 		= get_param((in_channels, out_channels))
		self.loop_rel 		= get_param((1, in_channels))

		self.w_at_in = get_param((2*out_channels, 1))
		self.w_at_out = get_param((2*out_channels, 1))

		self.drop		= torch.nn.Dropout(self.p.dropout)
		self.bn			= torch.nn.BatchNorm1d(out_channels)
		self.num_ent = 0

		if self.p.bias: self.register_parameter('bias', Parameter(torch.zeros(out_channels)))

	def forward(self, x, edge_index, edge_type, rel_embed): 
		if self.device is None:
			self.device = edge_index.device

		rel_embed = torch.cat([rel_embed, self.loop_rel], dim=0)
		num_edges = edge_index.size(1) // 2
		self.num_ent   = x.size(0)

		self.in_index, self.out_index = edge_index[:, :num_edges], edge_index[:, num_edges:]
		self.in_type,  self.out_type  = edge_type[:num_edges], 	 edge_type [num_edges:]
		self.loop_index  = torch.stack([torch.arange(self.num_ent), torch.arange(self.num_ent)]).to(self.device)
		self.loop_type   = torch.full((self.num_ent,), rel_embed.size(0)-1, dtype=torch.long).to(self.device)

		# self.in_norm     = self.compute_norm(self.in_index,  num_ent)
		# self.out_norm    = self.compute_norm(self.out_index, num_ent)
		
		# loop_res	= self.propagate('add', self.loop_index, x=x, edge_type=self.loop_type,
		# 							 rel_embed=rel_embed, edge_norm=None, mode='loop')
		# in_res		= self.propagate('add', self.in_index,   x=x, edge_type=self.in_type,
		# 	   						 rel_embed=rel_embed, edge_norm=self.in_norm, 	mode='in')
		# out_res		= self.propagate('add', self.out_index,  x=x, edge_type=self.out_type,  
		# 							 rel_embed=rel_embed, edge_norm=self.out_norm,	mode='out')
		loop_res	= self.propagate('add', self.loop_index, x=x, edge_type=self.loop_type,
									 rel_embed=rel_embed, gat=False, mode='loop', loop_res = None)
		in_res		= self.propagate('add', self.in_index,   x=x, edge_type=self.in_type,
			   						 rel_embed=rel_embed, gat=True, mode='in', loop_res = loop_res)
		out_res		= self.propagate('add', self.out_index,  x=x, edge_type=self.out_type,  
									 rel_embed=rel_embed, gat=True,	mode='out', loop_res = loop_res)
		out		= self.drop(in_res)*(1/3) + self.drop(out_res)*(1/3) + loop_res*(1/3)

		if self.p.bias: out = out + self.bias
		out = self.bn(out)

		return self.act(out), torch.matmul(rel_embed, self.w_rel)[:-1]		# Ignoring the self loop inserted

	def rel_transform(self, ent_embed, rel_embed):
		if   self.p.opn == 'corr': 	trans_embed  = ccorr(ent_embed, rel_embed)
		elif self.p.opn == 'sub': 	trans_embed  = ent_embed - rel_embed
		elif self.p.opn == 'mult': 	trans_embed  = ent_embed * rel_embed
		else: raise NotImplementedError

		return trans_embed

	def message(self, x_j, edge_type, rel_embed, gat, mode, loop_res, edge_index):
		weight 	= getattr(self, 'w_{}'.format(mode))
		rel_emb = torch.index_select(rel_embed, 0, edge_type)
		xj_rel  = self.rel_transform(x_j, rel_emb)
		out	= torch.mm(xj_rel, weight)
		if gat:
			gat_coef = self.compute_gat(edge_index, self.num_ent, out,loop_res,mode)
			return out*gat_coef.view(-1,1)
		return out

	def update(self, aggr_out):
		return aggr_out

	def compute_norm(self, edge_index, num_ent):
		row, col	= edge_index
		edge_weight 	= torch.ones_like(row).float()
		deg		= scatter_add( edge_weight, row, dim=0, dim_size=num_ent)	# Summing number of weights of the edges
		deg_inv		= deg.pow(-0.5)							# D^{-0.5}
		deg_inv[deg_inv	== float('inf')] = 0
		norm		= deg_inv[row] * edge_weight * deg_inv[col]			# D^{-0.5}
		return norm

	def compute_gat(self, edge_index, num_ent, x_j, loop_res, mode):
		weight = getattr(self, 'w_at_{}'.format(mode))
		row, col	= edge_index
		x_i = torch.index_select(loop_res, 0, row)

		cat = torch.cat((x_i, x_j), dim=1)
		unnorm_coef = cat.mm(weight)
		coef = torch.clamp(torch.exp(self.leakyrelu(unnorm_coef).squeeze()), min=0.0, max=1.0)
		row_sum = scatter_add(coef, row, dim=0, dim_size=num_ent)

		row_sum[row_sum == 0.0] = 1e-12
		return coef.div(row_sum[row])

	def __repr__(self):
		return '{}({}, {}, num_rels={})'.format(
			self.__class__.__name__, self.in_channels, self.out_channels, self.num_rels)
