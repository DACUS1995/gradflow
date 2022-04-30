import numpy as np
from sklearn import datasets

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


def train(model:Module, dataset:Dataset):
    optimizer = NaiveSGD(model.parameters)

    for X, y in dataset:
        pred_y = model(X)

        loss = (pred_y - y) @ (pred_y - y)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"Loss: {loss.data}")



def main():
    model = build_model()
    dataset = get_dataset()

    train(model, dataset)


if __name__ == "__main__":
    main()