"""
DESCRIPTION: models, optimizers and losses.
AUTHOR: Pablo Ferri-Borredà
DATE: 01/07/22
"""

# MODULES IMPORT
from torch import Tensor, tensor
from torch.cuda import is_available, device_count
from torch.nn import Module, CrossEntropyLoss, DataParallel, Softmax
from torch.optim import Optimizer, AdamW
from transformers import DistilBertForSequenceClassification

from hyperpars import HyperparamContainer

# COMMON SETTINGS
NUMBER_CLASSES = 2


# MODEL ARCHITECTURE
# DistilBERT based classifier
class TextClassifierDistilBERT(Module):
    # CLASS ATTRIBUTES
    # Device
    gpu = True if is_available() else False

    # INITIALIZATION
    def __init__(self) -> None:
        # Parent method call
        super().__init__()

        # DistilBERT model
        self.distilbert = DistilBertForSequenceClassification.from_pretrained('distilbert-base-multilingual-cased',
                                                                              num_labels=NUMBER_CLASSES)

        # Output activation function
        self.output_activation_function = Softmax(dim=1)

    # FORWARD
    def forward(self, indexes_attention_tensor: Tensor) -> Tensor:
        # Individual tensors extraction
        indexes_tensor = indexes_attention_tensor[:, :, 0]
        attention_tensor = indexes_attention_tensor[:, :, 1]

        # Logits obtention
        logits = self.distilbert.forward(input_ids=indexes_tensor, attention_mask=attention_tensor).logits

        # Probabilities obtention
        yhat = self.output_activation_function(logits)

        # Output
        return yhat


# MODEL DEFINITION
def define_model(model_identifier: str):
    # Model definition
    if model_identifier == 'TextClassifierDistilBERT':
        model = TextClassifierDistilBERT()
    else:
        raise ValueError('Unrecognized model identifier.')

    # Model parallelization in multiple GPUs (if possible)
    if device_count() > 1:
        model = DataParallel(model)

    # Output
    return model


# OPTIMIZER
def define_optimizer(model: Module, hyperparam_container: HyperparamContainer) -> Optimizer:
    # Learning rate extraction
    learning_rate = hyperparam_container.get_value('learning_rate')

    # Optimizer definition
    optimizer = AdamW(
        model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, amsgrad=False)

    # Output
    return optimizer


# LOSS FUNCTION
def define_loss_function(hyperparam_container: HyperparamContainer) -> CrossEntropyLoss:
    # Class weighting flag
    class_weighting = hyperparam_container.get_value('class_weighting')

    # Loss function class weighted
    if class_weighting:
        # Weights definition
        class_weights = tensor(hyperparam_container.get_value('class_weights'))
        if is_available():
            class_weights = class_weights.cuda()

        # Loss definition
        loss_function = CrossEntropyLoss(weight=class_weights)

    # Loss function without class weighting
    else:
        # Definition
        loss_function = CrossEntropyLoss()

    # Output
    return loss_function
