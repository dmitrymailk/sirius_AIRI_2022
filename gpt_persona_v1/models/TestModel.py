import torch
from torch import nn


class TestModel(nn.Module):
    def __init__(self, n_classes=None):
        super(TestModel, self).__init__()
        self.lin = nn.Linear(32, n_classes)
        self.n_classes = n_classes

    def forward(self, x):
        batch_size, sequence_length = x.shape[:2]
        hidden_states = torch.rand(
            (batch_size, sequence_length, 32), dtype=torch.float32, requires_grad=True
        )
        logits = self.lin(hidden_states)

        return logits

    def test(self):
        print("testt1")
