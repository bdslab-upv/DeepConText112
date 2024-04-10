"""
DESCRIPTION: main script to train and validate models to assess covariate shifts.
AUTHOR: Pablo Ferri-Borredà
DATE: 19/07/22
"""

# MODULES IMPORT
import sklearn.metrics as skmet
from numpy import argmax, concatenate
from numpy import ndarray
from torch import Tensor, no_grad
from torch.cuda import is_available
from torch.nn import Module, CrossEntropyLoss, Softmax
from torch.optim import AdamW
from tqdm import tqdm
from transformers import DistilBertForSequenceClassification


# TRAINING AND TESTING
# Iterative training and testing
def train_test(data_loaders: dict) -> dict:
    # Memory allocation
    results = dict()

    # Device definition
    device = 'cuda:0' if is_available() else 'cpu'

    # Loss function definition
    loss_function = CrossEntropyLoss()

    # Number epochs
    number_epochs = 3

    # Iteration across experiments
    for experiment_key, loaders_train_test in tqdm(data_loaders.items(), colour='green', position=0):
        # print('Experiment ' + str(experiment_key))

        # Train and test loaders extraction
        loader_train = loaders_train_test['train']
        loader_test = loaders_train_test['test']

        # Model initialization
        model = TextClassifierDistilBERT()

        # Device
        model.to(device)

        # Optimizer definition
        optimizer = AdamW(model.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01,
                          amsgrad=False)

        # Training mode enabling
        model.train()

        # Iteration across epochs
        for epoch in range(number_epochs):
            # print('Epoch ' + str(epoch))

            # Iteration across batches
            for batch_train, data_batch_train in enumerate(tqdm(loader_train, colour='cyan', position=0)):
                indexes_train = data_batch_train['indexes'].to(device)
                attention_mask_train = data_batch_train['attention_mask'].to(device)
                labels_train = data_batch_train['labels'].to(device)

                predictions_train = model.forward(indexes=indexes_train, attention_mask=attention_mask_train)

                loss = loss_function(predictions_train, labels_train)

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

        # Testing
        model.eval()

        labels_list = []
        outputs_scores_list, outputs_saturated_list = [], []

        with no_grad():
            for batch_test, data_batch_test in enumerate(loader_test):
                indexes_test = data_batch_test['indexes'].to(device)
                attention_mask_test = data_batch_test['attention_mask'].to(device)
                labels_test = data_batch_test['labels'].to(device)

                predictions_test = model.forward(indexes=indexes_test, attention_mask=attention_mask_test)

                outputs_scores_numpy = _tensor2array(predictions_test)
                labels_numpy = _tensor2array(labels_test)

                outputs_saturated_numpy = argmax(outputs_scores_numpy, axis=1)
                labels_saturated_numpy = argmax(labels_numpy, axis=1)

                labels_list.append(labels_saturated_numpy)
                outputs_scores_list.append(outputs_scores_numpy)
                outputs_saturated_list.append(outputs_saturated_numpy)

        # Concatenation
        labels_concatenated = concatenate(labels_list, axis=0)
        outputs_scores_concatenated = concatenate(outputs_scores_list, axis=0)
        outputs_saturated_concatenated = concatenate(outputs_saturated_list, axis=0)

        # Metrics calculation
        index2class_map = {0: str(experiment_key[0]), 1: str(experiment_key[1])}

        metrics_presatur = get_presaturation_classification_metrics(
            label_true=labels_concatenated, label_scores=outputs_scores_concatenated, index2class_map=index2class_map)
        metrics_satur = get_postsaturation_classification_metrics(
            label_true=labels_concatenated, label_predicted=outputs_saturated_concatenated,
            index2class_map=index2class_map)

        # Arrangement
        results[experiment_key] = {'presaturation': metrics_presatur, 'postsaturation': metrics_satur}

    # Output
    return results


# Tensor to array casting
def _tensor2array(tensor: Tensor):
    if tensor.is_cuda:
        try:
            return _tensor2array_gpu(tensor)
        except:
            return _tensor2array_detach_gpu(tensor)
    else:
        try:
            return _tensor2array_cpu(tensor)
        except:
            return _tensor2array_detach_cpu(tensor)


def _tensor2array_cpu(tensor: Tensor):
    return tensor.numpy()


def _tensor2array_gpu(tensor: Tensor):
    return tensor.cpu().numpy()


def _tensor2array_detach_cpu(tensor: Tensor):
    return tensor.detach().numpy()


def _tensor2array_detach_gpu(tensor: Tensor):
    return tensor.cpu().detach().numpy()


# Text classifier
class TextClassifierDistilBERT(Module):
    # INITIALIZATION
    def __init__(self) -> None:
        # Parent method call
        super().__init__()

        # DistilBERT model
        self.distilbert = DistilBertForSequenceClassification.from_pretrained('distilbert-base-multilingual-cased',
                                                                              num_labels=2)

        # Output activation function
        self.output_activation_function = Softmax(dim=1)

    # FORWARD
    def forward(self, indexes: Tensor, attention_mask: Tensor) -> Tensor:
        # Logits obtention
        logits = self.distilbert.forward(input_ids=indexes, attention_mask=attention_mask).logits

        # Probabilities obtention
        yhat = self.output_activation_function(logits)

        # Output
        return yhat


# SINGLE-LABEL PRE-SATURATION CLASSIFICATION METRICS
def get_presaturation_classification_metrics(label_true: ndarray, label_scores: ndarray,
                                             index2class_map: dict) -> dict:
    # Memory allocation
    metrics = dict()

    # Metrics calculation
    # memory allocation
    auc_classes = []  # area under curve per class

    # single-class
    for index, class_ in index2class_map.items():
        # class identifier generation
        class_idf = class_.upper()
        # binarization and extraction of scores per class
        if len(label_true.shape) == 1:
            label_true_class = label_true == index
        else:  # one-hot encoding
            label_true_class = label_true[:, index]
        label_true_class = label_true_class.astype(int)
        label_scores_class = label_scores[:, index]
        # area under curve per class calculation
        try:
            auc_class = skmet.roc_auc_score(label_true_class, label_scores_class)
        except:
            auc_class = 0
            print('Problem calculating area under curve.')
        # arrangement
        auc_classes.append(auc_class)
        metrics['AUC_' + class_idf] = auc_class

    # multi-class
    # area under curve
    metrics['AUC_MACRO'] = sum(auc_classes) / len(auc_classes)
    # cross-entropy loss
    try:
        metrics['LOGLOSS'] = skmet.log_loss(label_true, label_scores)
    except:
        metrics['LOGLOSS'] = 1
        print('Problem calculating logloss.')

    # Output
    return metrics


# SINGLE-LABEL POST-SATURATION CLASSIFICATION METRICS
def get_postsaturation_classification_metrics(label_true: ndarray, label_predicted: ndarray,
                                              index2class_map: dict) -> dict:
    # Memory allocation
    metrics = dict()

    # Metrics calculation
    # single-class
    for index, class_ in index2class_map.items():
        # class identifier generation
        class_idf = class_.upper()
        # binarization
        label_true_binarized = label_true == index
        label_predicted_binarized = label_predicted == index
        # recall
        metrics['RECALL_' + class_idf] = skmet.recall_score(
            label_true_binarized, label_predicted_binarized, average='binary')
        # precision
        metrics['PRECISION_' + class_idf] = skmet.precision_score(
            label_true_binarized, label_predicted_binarized, average='binary')
        # f1_score
        metrics['F1-SCORE_' + class_idf] = skmet.f1_score(
            label_true_binarized, label_predicted_binarized, average='binary')

    # multi-class
    # accuracy
    metrics['ACCURACY'] = skmet.accuracy_score(label_true, label_predicted)
    # recall
    metrics['RECALL_MACRO'] = skmet.recall_score(label_true, label_predicted, average='macro')
    metrics['RECALL_MICRO'] = skmet.recall_score(label_true, label_predicted, average='micro')
    metrics['RECALL_WEIGHTED'] = skmet.recall_score(label_true, label_predicted, average='weighted')
    # precision
    metrics['PRECISION_MACRO'] = skmet.precision_score(label_true, label_predicted, average='macro')
    metrics['PRECISION_MICRO'] = skmet.recall_score(label_true, label_predicted, average='micro')
    metrics['PRECISION_WEIGHTED'] = skmet.recall_score(label_true, label_predicted, average='weighted')
    # f1-score
    metrics['F1-SCORE_MACRO'] = skmet.f1_score(label_true, label_predicted, average='macro')
    metrics['F1-SCORE_MICRO'] = skmet.f1_score(label_true, label_predicted, average='micro')
    metrics['F1-SCORE_WEIGHTED'] = skmet.f1_score(label_true, label_predicted, average='weighted')

    # Output
    return metrics
