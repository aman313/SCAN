import torch
from torch import nn
from functools import reduce

class RMCCell(nn.Module):
    
    def __init__(self,mem_slots,head_size,num_heads=1,num_blocks=1,forget_bias=1.0,input_bias=0.0,gate_style='unit',attention_mlp_layers=2,key_size=None,device="cpu"):
        super(RMCCell, self).__init__()
        self._mem_slots = mem_slots
        self._head_size = head_size
        self._num_heads = num_heads
        self._mem_size = self._head_size * self._num_heads
        self._device = device
    
        if num_blocks < 1:
            raise ValueError('num_blocks must be >= 1. Got: {}.'.format(num_blocks))
        self._num_blocks = num_blocks
    
        self._forget_bias = forget_bias
        self._input_bias = input_bias
    
        if gate_style not in ['unit', 'memory', None]:
            raise ValueError(
              'gate_style must be one of [\'unit\', \'memory\', None]. Got: '
              '{}.'.format(gate_style))
        self._gate_style = gate_style
    
        if attention_mlp_layers < 1:
            raise ValueError('attention_mlp_layers must be >= 1. Got: {}.'.format(
              attention_mlp_layers))
        self._attention_mlp_layers = attention_mlp_layers
        self._key_size = key_size if key_size else self._head_size
        
    def initial_state(self,batch_size):
        init_state = torch.stack(batch_size*[torch.eye(self._mem_slots)]).to(self._device)
        # Pad the matrix with zeros.
        if self._mem_size > self._mem_slots:
            difference = self._mem_size - self._mem_slots
            pad = torch.zeros(batch_size, self._mem_slots, difference).to(self._device)
            init_state = torch.cat([init_state, pad], -1)
        # Truncation. Take the first `self._mem_size` components.
        elif self._mem_size < self._mem_slots:
            init_state = init_state[:, :, :self._mem_size]
        return init_state
    
    def _multihead_attention(self, memory):
        """Perform multi-head attention from 'Attention is All You Need'.
        Implementation of the attention mechanism from
        https://arxiv.org/abs/1706.03762.
        Args:
          memory: Memory tensor to perform attention on.
        Returns:
          new_memory: New memory tensor.
        """
        key_size = self._key_size
        value_size = self._head_size
        
        qkv_size = 2 * key_size + value_size
        total_size = qkv_size * self._num_heads  # Denote as F.
        qkv = torch.nn.Linear(memory.shape[2],total_size).to(self._device)(memory)
        qkv =torch.nn.modules.normalization.LayerNorm(qkv.shape[-1]).to(self._device)(qkv)
        
        mem_slots = memory.shape[1]  # Denoted as N.
        qkv_reshape = qkv.reshape(qkv.shape[0],mem_slots,self._num_heads,qkv_size)
        qkv_transponse = torch.transpose(qkv_reshape,1,2)
        q,k,v = torch.split(qkv_transponse,[key_size,key_size,value_size],-1)
        q *= qkv_size**-0.5
        dot_product = torch.matmul(q,torch.transpose(k,2,3))
        weights = torch.nn.Softmax()(dot_product)
        output = torch.matmul(weights,v)
        output_transpose = torch.transpose(output,1,2)
        new_memory = output_transpose.contiguous().view(output_transpose.shape[0],output_transpose.shape[1],output_transpose.shape[2]*output_transpose.shape[3])
        return new_memory

    def _calculate_gate_size(self):
        """Calculate the gate size from the gate_style.
        Returns:
          The per sample, per head parameter size of each gate.
        """
        if self._gate_style == 'unit':
            return self._mem_size
        elif self._gate_style == 'memory':
            return 1
        else:  # self._gate_style == None
            return 0

    def _create_gates(self, inputs, memory):
        """Create input and forget gates for this step using `inputs` and `memory`.
        Args:
          inputs: Tensor input.
          memory: The current state of memory.
        Returns:
          input_gate: A LSTM-like insert gate.
          forget_gate: A LSTM-like forget gate.
        """
        # We'll create the input and forget gates at once. Hence, calculate double
        # the gate size.
        num_gates = 2 * self._calculate_gate_size()
        memory = torch.nn.Tanh()(memory)
        inputs = inputs.contiguous().view(inputs.shape[0],reduce(lambda x, y: x*y, inputs.shape[1:]))
        gate_inputs = torch.nn.Linear(inputs.shape[-1],num_gates).to(self._device)(inputs)
        gate_memory = torch.nn.Linear(memory.shape[-1],num_gates).to(self._device)(memory)
        gates = torch.split(gate_memory + gate_inputs, split_size_or_sections=int(gate_memory.shape[2]/2), dim=2)
        input_gate, forget_gate = gates
        input_gate = torch.nn.Sigmoid()(input_gate + self._input_bias)
        forget_gate = torch.nn.Sigmoid()(forget_gate + self._forget_bias)
        return input_gate, forget_gate
    
    def _attend_over_memory(self, memory):
        """Perform multiheaded attention over `memory`.
        Args:
          memory: Current relational memory.
        Returns:
          The attended-over memory.
        """
        
        def _attn_mlp_builder(num_layers,num_hidden):
            def _apply(input):
                in_size =  input.shape[-1]
                for _ in range(num_layers):
                    ll = torch.nn.Linear(in_size,num_hidden).to(self._device)
                    in_size = num_hidden
                    output = ll(input)
                    input = output
                return output
            return _apply
        attention_mlp = _attn_mlp_builder(self._attention_mlp_layers, self._mem_size)
        
        for _ in range(self._num_blocks):
            attended_memory = self._multihead_attention(memory)
            # Add a skip connection to the multiheaded attention's input.
            memory = torch.nn.modules.normalization.LayerNorm(memory.shape[-1]).to(self._device)(memory + attended_memory)
            # Add a skip connection to the attention_mlp's input.
            memory = torch.nn.modules.normalization.LayerNorm(memory.shape[-1]).to(self._device)(attention_mlp(memory) + memory)
            
        return memory
 
    def forward(self,inputs,memory):
        inputs = inputs.contiguous().view(inputs.shape[0],reduce(lambda x, y: x*y, inputs.shape[1:]))
        inputs = torch.nn.Linear(inputs.shape[-1],self._mem_size).to(self._device)(inputs)
        inputs_reshape = inputs.unsqueeze(1)
        memory_plus_inputs = torch.cat([memory,inputs_reshape],dim=1)
        next_memory = self._attend_over_memory(memory_plus_inputs)
        n = inputs_reshape.shape[1]
        next_memory = next_memory[:, :-n, :]
        if self._gate_style == 'unit' or self._gate_style == 'memory':
            self._input_gate, self._forget_gate = self._create_gates(
                inputs_reshape, memory)
            next_memory = self._input_gate * torch.nn.Tanh()(next_memory)
            next_memory += self._forget_gate * memory
        output = next_memory.contiguous().view(next_memory.shape[0],reduce(lambda x, y: x*y, next_memory.shape[1:]))
        return output,next_memory


class RMC(nn.Module):
    def __init__(self, rmc_cell):
        super().__init__()
        self.rmc_cell = rmc_cell
        
    def forward(self, inputs,hidden=None):
        outputs = []
        hidden_vals = []
        #if hidden is None:
        hidden = self.rmc_cell.initial_state(inputs.shape[0])
        for x in torch.unbind(inputs, dim=1):
            hidden = self.rmc_cell(x, hidden)
            outputs.append(hidden[0].clone())
            hidden_vals.append(hidden[0])
            hidden = hidden[1]
        return torch.stack(outputs, dim=1),torch.stack(hidden_vals)    
    
if __name__ =='__main__':
    rmccell = RMCCell(3,2,2)
    input = torch.ones(3,3,5)
    rmc = RMC(rmccell)
    rmc(input,rmccell.initial_state(3))