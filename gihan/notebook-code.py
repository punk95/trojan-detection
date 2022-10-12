import torch
import os
import json
from tqdm.notebook import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
cudnn.benchmark = True  # fire on all cylinders
from sklearn.metrics import roc_auc_score, roc_curve
import sys

sys.path.insert(0, '..')






# CREATE DATASET CLASSES

class NetworkDatasetDetection(torch.utils.data.Dataset):
    def __init__(self, model_folder):
        super().__init__()
        model_paths = []
        model_paths.extend([os.path.join(model_folder, 'clean', x) \
                            for x in sorted(os.listdir(os.path.join(model_folder, 'clean')))])
        model_paths.extend([os.path.join(model_folder, 'trojan', x) \
                            for x in sorted(os.listdir(os.path.join(model_folder, 'trojan')))])
        labels = []
        data_sources = []
        for p in model_paths:
            with open(os.path.join(p, 'info.json'), 'r') as f:
                info = json.load(f)
                data_sources.append(info['dataset'])
            if p.split('/')[-2] == 'clean':
                labels.append(0)
            elif p.split('/')[-2] == 'trojan':
                labels.append(1)
            else:
                raise ValueError('unexpected path {}'.format(p))
        self.model_paths = model_paths
        self.labels = labels
        self.data_sources = data_sources
    
    def __len__(self):
        return len(self.model_paths)
    
    def __getitem__(self, index):
        return torch.load(os.path.join(self.model_paths[index], 'model.pt')), \
               self.labels[index], self.data_sources[index]

def custom_collate(batch):
    return [x[0] for x in batch], [x[1] for x in batch], [x[2] for x in batch]


# LOAD DATASET

dataset_path = '../datasets'
task = 'detection'
dataset = NetworkDatasetDetection(os.path.join(dataset_path, task, 'train'))

split = int(len(dataset) * 0.8)
rnd_idx = np.random.permutation(len(dataset))
train_dataset = torch.utils.data.Subset(dataset, rnd_idx[:split])
val_dataset = torch.utils.data.Subset(dataset, rnd_idx[split:])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True,
                                           num_workers=0, pin_memory=False, collate_fn=custom_collate)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                           num_workers=0, pin_memory=False, collate_fn=custom_collate)


#  CONSTRUCT MNDT NETWORK

data_sources = ['CIFAR-10', 'CIFAR-100', 'GTSRB', 'MNIST']
data_source_to_channel = {k: 1 if k == 'MNIST' else 3 for k in data_sources}
data_source_to_resolution = {k: 28 if k == 'MNIST' else 32 for k in data_sources}
data_source_to_num_classes = {'CIFAR-10': 10, 'CIFAR-100': 100, 'GTSRB': 43, 'MNIST': 10}

class MetaNetwork(nn.Module):
    def __init__(self, num_queries, num_classes=1):
        super().__init__()
        self.queries = nn.ParameterDict(
            {k: nn.Parameter(torch.rand(num_queries,
                                        data_source_to_channel[k],
                                        data_source_to_resolution[k],
                                        data_source_to_resolution[k])) for k in data_sources}
        )
        self.affines = nn.ModuleDict(
            {k: nn.Linear(data_source_to_num_classes[k]*num_queries, 32) for k in data_sources}
        )
        self.norm = nn.LayerNorm(32)
        self.relu = nn.ReLU(True)
        self.final_output = nn.Linear(32, num_classes)
    
    def forward(self, net, data_source):
        """
        :param net: an input network of one of the model_types specified at init
        :param data_source: the name of the data source
        :returns: a score for whether the network is a Trojan or not
        """
        query = self.queries[data_source]
        out = net(query)
        out = self.affines[data_source](out.view(1, -1))
        out = self.norm(out)
        out = self.relu(out)
        return self.final_output(out)

# TRAIN THE NETWORK

meta_network = MetaNetwork(10, num_classes=1).cuda().train()

num_epochs = 10
lr = 0.01
weight_decay = 0.
optimizer = torch.optim.Adam(meta_network.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs * len(train_dataset))

loss_ema = np.inf
for epoch in range(num_epochs):
    print("GGGG")
    pbar = tqdm(train_loader)
    pbar.set_description(f"Epoch {epoch + 1}")
    for i, (net, label, data_source) in enumerate(pbar):
        net = net[0]
        label = label[0]
        data_source = data_source[0]
        net.cuda().eval()
        
        out = meta_network(net, data_source)
        
        loss = F.binary_cross_entropy_with_logits(out, torch.FloatTensor([label]).unsqueeze(0).cuda())
        
        optimizer.zero_grad()
        loss.backward(inputs=list(meta_network.parameters()))
        optimizer.step()
        scheduler.step()
        for k in meta_network.queries.keys():
            meta_network.queries[k].data = meta_network.queries[k].data.clamp(0, 1)
        loss_ema = loss.item() if loss_ema == np.inf else 0.95 * loss_ema + 0.05 * loss.item()

        pbar.set_postfix(loss=loss_ema)

meta_network.eval()


# Evaluate the network


def evaluate(meta_network, loader):
    loss_list = []
    correct_list = []
    confusion_matrix = torch.zeros(2,2)
    all_scores = []
    all_labels = []
    
    for i, (net, label, data_source) in enumerate(tqdm(loader)):
        net[0].cuda().eval()
        with torch.no_grad():
            out = meta_network(net[0], data_source[0])
        loss = F.binary_cross_entropy_with_logits(out, torch.FloatTensor([label[0]]).unsqueeze(0).cuda())
        correct = int((out.squeeze() > 0).int().item() == label[0])
        loss_list.append(loss.item())
        correct_list.append(correct)
        confusion_matrix[(out.squeeze() > 0).int().item(), label[0]] += 1
        all_scores.append(out.squeeze().item())
        all_labels.append(label[0])
    
    return np.mean(loss_list), np.mean(correct_list), confusion_matrix, all_labels, all_scores

# 

loss, acc, cmat, _, _ = evaluate(meta_network, train_loader)
print(f'Train Loss: {loss:.3f}, Train Accuracy: {acc*100:.2f}')
print('Confusion Matrix:\n', cmat.numpy())


loss, acc, cmat, all_labels, all_preds = evaluate(meta_network, val_loader)
print(f'Val Loss: {loss:.3f}, Val Accuracy: {acc*100:.2f}')
print('Confusion Matrix:\n', cmat.numpy())


print(f'Val AUROC: {roc_auc_score(all_labels, all_preds):.3f}')
