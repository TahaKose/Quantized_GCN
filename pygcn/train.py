from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy
from models import GCN



import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
from models import GCN
from utils import load_data, accuracy
from quant import *


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='cora',
                    help='dataset.')
parser.add_argument('--early_stopping', type=int, default=10,
                    help='Number of epochs for early_stopping.')
parser.add_argument('--w_bit', type=int, default=32,
                    help='Number of weight quantization bits')
parser.add_argument('--f_bit', type=int, default=32,
                    help='Number of features quantization bits')
parser.add_argument('--a_bit', type=int, default=32,
                    help='Number of adjacency quantization bits')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data(args.dataset)

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout,
            nf_bit=args.f_bit,
            na_bit=args.a_bit)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train():
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    return (loss_train.item(),acc_train.item(),loss_val.item(),acc_val.item())


def test(quant=False):
    model.eval()
    output = model(features, adj, quant=quant)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

# Train model without quantization
print("Training without quantization...")
for epoch in range(args.epochs):
    t = time.time()
    losses_val = []
    loss_train, acc_train, loss_val, acc_val = train()
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train),
          'acc_train: {:.4f}'.format(acc_train),
          'loss_val: {:.4f}'.format(loss_val),
          'acc_val: {:.4f}'.format(acc_val),
          'time: {:.4f}s'.format(time.time() - t))
    losses_val.append(loss_val)
    if epoch > args.early_stopping and losses_val[-1] > np.mean(losses_val[-(args.early_stopping + 1):-1]):
        print("Early stopping...")
        break

# Save the model after training without quantization
torch.save(model.state_dict(), "model_no_quantization.pth")

# Testing without quantization
print("Testing without quantization...")
test(quant=False)

# Quantize model weights and activations
quantized_state_dict = model.state_dict()
for key in quantized_state_dict:
    if "weight" in key:
        quantized_state_dict[key] = quantize_weights(quantized_state_dict[key], num_bits=args.w_bit)

# Print original and quantized weights
for key in model.state_dict():
    if "weight" in key:
        original_weight = model.state_dict()[key]
        quantized_weight = quantized_state_dict[key]
        print(f"Layer: {key}")
        print("Original Weight:")
        print(original_weight)
        print("Quantized Weight:")
        print(quantized_weight)
        print("=" * 30)

# Load quantized model weights and activations
model.load_state_dict(quantized_state_dict)

# Testing with quantized weights and activations
print("Testing with quantized weights and activations...")
test(quant=True)
