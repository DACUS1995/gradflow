from typing import Dict

import numpy as np
from sklearn import datasets
from matplotlib import pyplot as plt

from gradflow.data.dataset import Dataset
from gradflow.grad_engine import Variable
from gradflow.layers.linear import Linear
from gradflow.layers.module import Module
from gradflow.optimizers import NaiveSGD


def get_dataset() -> Dataset:
    X, y = datasets.make_regression(
        n_samples=100,
        n_features=1,
        n_informative=1,
        noise=10,
        coef=False,
        random_state=0
    )

    dataset = Dataset(X, y, batch_size=1)
    return dataset


def build_model() -> Module:
    class ModuleExample(Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = self.add_module(Linear(1, 1))
        
        def forward(self, input: Variable) -> Variable:
            out = self.linear(input=input)
            return out

    return ModuleExample()


def train(model:Module, dataset:Dataset, config:Dict):
    optimizer = NaiveSGD(model.parameters, lr=config["lr"])
    training_loss = []
    images = []

    for epoch in range(config["epochs"]):
        epoch_loss = .0

        for X, y in dataset:
            pred_y = model(X)

            loss = (pred_y - y) @ (pred_y - y)
            epoch_loss += loss.data.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss /= len(dataset)
        training_loss.append(epoch_loss)
        print(f"Epoch {epoch} | Loss: {epoch_loss}")

    plt.title("Training loss")
    plt.plot(training_loss)
    plt.show()


def main():
    model = build_model()
    dataset = get_dataset()
    config = {
        "epochs": 10,
        "lr": 0.001
    }

    train(
        model, 
        dataset, 
        config
    )


if __name__ == "__main__":
    main()
