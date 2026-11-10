"""Configure dropout before constructing a pretrained seq2seq model."""
import math


def configure_dropout(config, rate, scope='model'):
    """Keep the old single-field behaviour unless all seq2seq dropout is requested.

    BART uses functional attention/activation dropout in addition to residual
    dropout. Its classifier dropout is unused by conditional generation, and
    LayerDrop (stochastic layer removal) is a separate hyperparameter.
    """
    if not isinstance(rate, (int, float)) or not math.isfinite(rate) or not 0 <= rate < 1:
        raise ValueError('dropout must be a finite probability in [0, 1)')
    if scope not in ('model', 'all'):
        raise ValueError('dropout_scope must be model or all')
    if config.model_type == 'bart':
        fields = ('dropout', 'attention_dropout', 'activation_dropout') if scope == 'all' else ('dropout',)
    elif config.model_type == 't5':
        fields = ('dropout_rate',)
    else:
        raise ValueError(f'unsupported model type {config.model_type}')
    for name in fields:
        setattr(config, name, rate)
    return {name: getattr(config, name) for name in fields}
