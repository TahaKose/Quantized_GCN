# Quantized_GCN
PyGCN quantization with DoReFa-Net 

This quantization operation quantizes the weight, feature and adjacency matrices by w_bit, f_bit and a_bit respectively.

Might be required
pip install scipy

Hyperparameters
Hidden size = 64, Epoch size = 200, Learning rate = 0.01, 

python /pygcn/pygcn/train.py --dataset cora --early_stopping 10 --f_bit 32 --a_bit 32 --w_bit 32

python /pygcn/pygcn/train.py --dataset cora --early_stopping 10 --f_bit 32 --a_bit 16 --w_bit 16

python /pygcn/pygcn/train.py --dataset cora --early_stopping 10 --f_bit 32 --a_bit 8 --w_bit 8

python /pygcn/pygcn/train.py --dataset cora --early_stopping 10 --f_bit 32 --a_bit 4 --w_bit 4

python /pygcn/pygcn/train.py --dataset pubmed --early_stopping 10 --f_bit 32 --a_bit 32 --w_bit 32

python /pygcn/pygcn/train.py --dataset pubmed --early_stopping 10 --f_bit 32 --a_bit 16 --w_bit 16

python /pygcn/pygcn/train.py --dataset pubmed --early_stopping 10 --f_bit 32 --a_bit 8 --w_bit 8

python /pygcn/pygcn/train.py --dataset pubmed --early_stopping 10 --f_bit 32 --a_bit 4 --w_bit 4

python /pygcn/pygcn/train.py --dataset citeseer --early_stopping 10 --f_bit 32 --a_bit 32 --w_bit 32

python /pygcn/pygcn/train.py --dataset citeseer --early_stopping 10 --f_bit 32 --a_bit 16 --w_bit 16

python /pygcn/pygcn/train.py --dataset citeseer --early_stopping 10 --f_bit 32 --a_bit 8 --w_bit 8

python /pygcn/pygcn/train.py --dataset citeseer --early_stopping 10 --f_bit 32 --a_bit 4 --w_bit 4
