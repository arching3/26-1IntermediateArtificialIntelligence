import torch.nn as nn
from models.activations import get_activations


class MLP(nn.Module):
    def __init__(self, input_shape, hidden_sizes, num_classes, activation, dropout_p, use_batch_norm ):
        super().__init__()
        self.activations_name = activation
        self.dropout_p = dropout_p
        self.use_batch_norm = use_batch_norm
        input_layer = self.build_mlp_block(input_shape, hidden_sizes[0])
        hidden_layers = [self.build_mlp_block(hidden_sizes[i], hidden_sizes[i+1]) 
                              for i in range(len(hidden_sizes)-1)]
        output_layer = nn.Linear(hidden_sizes[-1], num_classes)
        self.model = nn.Sequential(
            input_layer,
            *hidden_layers,
            output_layer,
        )
    def build_mlp_block(self, _is, _os):
        linear = nn.Linear(_is, _os)
        if self.use_batch_norm:
            norm = nn.BatchNorm1d(_os)
        else:
            norm = get_activations('x')()
        if self.dropout_p > 0.0:
            dropout = nn.Dropout(self.dropout_p)
        else:
            dropout = get_activations('x')()
        activation_f = get_activations(self.activations_name)()
        sequential = nn.Sequential(
            linear,
            norm,
            activation_f,
            dropout,
        )
        return sequential

    def forward(self, x):
        return self.model(x)

def build_model(input_shape, num_classes, model_config):
    return MLP(
        input_shape=input_shape[0],
        hidden_sizes=model_config["hidden_sizes"],
        num_classes=num_classes,
        activation=model_config["activation"],
        dropout_p=model_config["dropout_p"],
        use_batch_norm=model_config["use_batch_norm"],
    )

