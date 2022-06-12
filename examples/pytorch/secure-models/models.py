import math

import torch
import torch.nn.functional as F

import crypten
import crypten.nn as cnn

from utils import softmax_2RELU, activation_quad

class Bert(cnn.Module):
    def __init__(self, config):
        super(Bert, self).__init__()
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = cnn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
    
    def forward(self, input_ids):
        output = self.embeddings(input_ids)
        for _, layer in enumerate(self.encoder):
            output = layer(output)
        return output

class BertEmbeddings(cnn.Module):
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = cnn.Linear(config.vocab_size, config.hidden_size)
        self.position_embeddings = cnn.Linear(config.max_position_embeddings, config.hidden_size)
        self.LayerNorm = cnn.BatchNorm2d(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = cnn.Dropout(config.hidden_dropout_prob)
        #self.position_ids = F.one_hot(torch.arange(config.max_position_embeddings)).item()
        #F.one_hot(torch.arange(config.max_position_embeddings))      
        self.config = config

    def forward(self, input_ids):
        embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings.weight[input_ids.shape[1]-1, :]
        position_embeddings = position_embeddings.repeat(input_ids.shape[0],1,1)
        embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertLayer(cnn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.config = config
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
 
    def forward(self, hidden_states):
        attention_output = self.attention(hidden_states)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
        
class BertAttention(cnn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
    
    def forward(self, hidden_states):
        self_output = self.self(hidden_states)
        attention_output = self.output(self_output, hidden_states)
        return attention_output 

class BertSelfAttention(cnn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.attention_head_size = self.hidden_size // self.num_attention_heads

        self.query = cnn.Linear(self.hidden_size, self.hidden_size)
        self.key = cnn.Linear(self.hidden_size, self.hidden_size)
        self.value = cnn.Linear(self.hidden_size, self.hidden_size)

        self.dropout = cnn.Dropout(config.attention_probs_dropout_prob)
        if config.softmax_act == "softmax":
            self.smax = cnn.Softmax(dim=-1)
        elif config.softmax_act == "softmax_2RELU":
            self.smax = softmax_2RELU(dim=-1)
        else:
            raise ValueError(f"softmax type {config.softmax_act} not implemented.")

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        attention_scores = query_layer.matmul(key_layer.transpose(-1, -2))
        print(attention_scores.shape)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = self.smax(attention_scores)

        attention_probs = self.dropout(attention_probs)
        context_layer = attention_probs.matmul(value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3)#.contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.reshape(new_context_layer_shape)
        
        return context_layer

class BertSelfOutput(cnn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = cnn.Linear(config.hidden_size, config.hidden_size)
        # using batchnorm here, crypten has not implemented LayerNorm
        self.LayerNorm = cnn.BatchNorm2d(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = cnn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # residual connection here
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertIntermediate(cnn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = cnn.Linear(config.hidden_size, config.intermediate_size)
        if config.hidden_act == "relu":
            self.intermediate_act_fn = cnn.ReLU()
        elif config.hidden_act == "quad":
            self.intermediate_act_fn = activation_quad()
        else:
            raise ValueError(f"activation type {config.hidden_act} not implemented")

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(cnn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = cnn.Linear(config.intermediate_size, config.hidden_size)
        # using batchnorm here, crypten has not implemented LayerNorm
        self.LayerNorm = cnn.BatchNorm2d(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = cnn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # residual connection
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
