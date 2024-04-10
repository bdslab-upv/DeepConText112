"""
DESCRIPTION: hyperparameters of models and strategies.
AUTHOR: Pablo Ferri-Borredà
DATE: 04/07/22
"""

# MODULES IMPORT
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Union


# ITEM CLASS DEFINITION
class Item(ABC):

    # INITIALIZATION
    @abstractmethod
    def __init__(self, identifier: Union[str, tuple]) -> None:
        self._identifier = identifier

    # EXTERNAL ATTRIBUTE ACCESS AND EDITION CONTROL
    @property
    def identifier(self) -> str:
        return self._identifier


# HYPERPARAMETER CLASS DEFINITION
class Hyperparameter(Item):

    # INITIALIZATION
    def __init__(self, identifier: str, default: Union[bool, int, float, str, list], tune: bool = False,
                 search_space: Union[None, tuple] = None):
        super().__init__(identifier)

        if default is None:
            raise ValueError

        if type(tune) is not bool:
            raise TypeError

        if tune:
            if type(search_space) is not tuple:
                raise TypeError

        self._default = default
        self._tune = tune
        self._search_space = search_space
        self.value = None if tune else self._default

    # EXTERNAL ATTRIBUTE ACCESS AND EDITION CONTROL
    @property
    def default(self):
        return self._default

    @property
    def tune(self):
        return self._tune

    @property
    def search_space(self):
        return self._search_space


# STATIC CONTAINER CLASS DEFINITION
class StaticContainer(ABC):

    # INITIALIZATION
    @abstractmethod
    def __init__(self) -> None:
        self._content = {}
        self._identifiers = []

    # EXTERNAL ATTRIBUTE ACCESS AND EDITION CONTROL
    @property
    def content(self) -> dict:
        return self._content

    @property
    def identifiers(self) -> list:
        return self._identifiers

    # ITEM GETTING
    def get_item(self, item_identifier: str) -> Item:
        return self._content[item_identifier]

    # ITEMS ADDITION
    # Multiple items at once
    def add_items_batch(self, items: Iterable) -> None:
        # Type checking
        if not isinstance(items, Iterable):
            raise TypeError('Items must be passed as an Iterable object.')

        # Items addition
        for item in items:
            self.add_item(item)

    # Single item
    def add_item(self, item: Item) -> None:
        # Inputs checking
        # type
        if not isinstance(item, Item):
            raise TypeError('Item must be an instance of Item or one of its subclasses.')

        # consistency
        if item.identifier in self._content.keys():
            raise ValueError('Two items cannot have the same identifier. Conflicting identifier: ' + item.identifier)

        # Addition
        self._content[item.identifier] = item
        self._identifiers.append(item.identifier)


# HYPERPARAMETER CONTAINER CLASS DEFINITION
class HyperparamContainer(StaticContainer):

    def __init__(self):
        super().__init__()

    def get_value(self, hyperparam_identifier: str):
        hyperparam = self._content[hyperparam_identifier]

        return hyperparam.value

    def set_value(self, hyperparam_identifier: str, value) -> None:
        self._content[hyperparam_identifier].value = value

    def add_hyperparams_batch(self, hyperparams: Iterable):
        super().add_items_batch(items=hyperparams)

    def add_hyperparam(self, hyperparam: Hyperparameter) -> None:
        if not isinstance(hyperparam, Hyperparameter):
            raise TypeError

        super().add_item(item=hyperparam)


# COMMON HYPERPARAMETERS
common = [
    # Class weighting
    Hyperparameter(identifier='class_weighting', default=True, tune=False),

    # Class weights
    Hyperparameter(identifier='class_weights', default=[0.4, 0.6], tune=False),
]

# SPECIFIC HYPERPARAMETERS
# JointTraining
joint_training = [
    # Learning rate
    Hyperparameter(identifier='learning_rate', default=0.00001, tune=False, search_space=(0.0001, 0.00001)),

    # Batch size
    Hyperparameter(identifier='batch_size_train', default=32, tune=False, search_space=(16, 32)),

    # Number epochs
    Hyperparameter(identifier='number_epochs', default=3, tune=False, search_space=(3, 4, 6))
]

# Cumulative
cumulative = [
    # Learning rate
    Hyperparameter(identifier='learning_rate', default=0.00001, tune=False, search_space=(0.0001, 0.00001)),

    # Batch size
    Hyperparameter(identifier='batch_size_train', default=32, tune=False, search_space=(16, 32)),

    # Number epochs
    Hyperparameter(identifier='number_epochs', default=3, tune=False, search_space=(3, 4, 6))
]

# SingleFineTuning
single_fine_tuning = [
    # Learning rate
    Hyperparameter(identifier='learning_rate', default=0.00001, tune=False, search_space=(0.0001, 0.00001)),

    # Batch size
    Hyperparameter(identifier='batch_size_train', default=16, tune=False, search_space=(16, 32)),

    # Number epochs
    Hyperparameter(identifier='number_epochs', default=4, tune=False, search_space=(3, 4, 6))
]

# ContinualFineTuning
continual_fine_tuning = [
    # Learning rate
    Hyperparameter(identifier='learning_rate', default=0.00001, tune=False, search_space=(0.0001, 0.00001)),

    # Batch size
    Hyperparameter(identifier='batch_size_train', default=32, tune=False, search_space=(16, 32)),

    # Number epochs
    Hyperparameter(identifier='number_epochs', default=3, tune=False, search_space=(3, 4, 6))
]

# Replay
replay = [
    # Learning rate
    Hyperparameter(identifier='learning_rate', default=0.00001, tune=False, search_space=(0.0001, 0.00001)),

    # Batch size
    Hyperparameter(identifier='batch_size_train', default=16, tune=False, search_space=(16, 32)),

    # Number epochs
    Hyperparameter(identifier='number_epochs', default=2, tune=False, search_space=(2, 3, 4)),

    # Memory size
    Hyperparameter(identifier='memory_size', default=16384, tune=False, search_space=(4096, 16384, 32768))
]

# Synaptic intelligence
synaptic_intelligence = [
    # Learning rate
    Hyperparameter(identifier='learning_rate', default=0.00001, tune=False, search_space=(0.0001, 0.00001)),

    # Batch size
    Hyperparameter(identifier='batch_size_train', default=32, tune=False, search_space=(16, 32)),

    # Number epochs
    Hyperparameter(identifier='number_epochs', default=3, tune=False, search_space=(3, 4, 6)),

    # Lambda
    Hyperparameter(identifier='lambda', default=0.1, tune=False, search_space=(0.1, 0.25, 0.4))
]

# HYPERPARAMETERS DICTIONARY
hyperparams_map = {
    # Joint
    'JointTraining': common + joint_training,

    # Cumulative
    'Cumulative': common + cumulative,

    # SingleFineTuning
    'SingleExperience': common + single_fine_tuning,

    # ContinualFineTuning
    'ContinualFineTuning': common + continual_fine_tuning,

    # Replay
    'Replay': common + replay,

    # Synaptic intelligence
    'SynapticIntelligence': common + synaptic_intelligence,

}


# HYPERPARAMETERS CONTAINER
def define_hyperparameters(strategy_identifier: str) -> HyperparamContainer:
    # Initialization
    hyperparam_container = HyperparamContainer()

    # Hyperparameters addition
    try:
        hyperparam_container.add_hyperparams_batch(hyperparams=hyperparams_map[strategy_identifier])
    except KeyError:
        print('Unrecognized strategy identifier.')

    # Output
    return hyperparam_container
