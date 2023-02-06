
import torch
from dataset.ssa.dataset_toy import sequenceDataset
from torch.utils.data import DataLoader
import tqdm

def collate(batch):
    (pasts_list, futures_list, num_agents) = zip(*batch)
    pasts = torch.cat(pasts_list)
    futures = torch.cat(futures_list)

    track = torch.cat((pasts, futures), 1)
    track_rel = torch.zeros(track.shape)
    track_rel[:, 1:] = track[:, 1:] - track[:, :-1]
    pasts_rel = track_rel[:, :20]
    futures_rel = track_rel[:, 20:]

    _len = num_agents
    return pasts, futures, pasts_rel, futures_rel, _len

len_past = 20
num_input = 2
batch_size = 32
print('loading dataset...')
data_train = sequenceDataset('dataset/ssa/data_train.npy', len_past, num_input)
data_test = sequenceDataset('dataset/ssa/test.npy', len_past, num_input)
loader_train = DataLoader(data_train, collate_fn=collate, batch_size=batch_size, num_workers=0,
                               shuffle=True)
loader_test = DataLoader(data_test, collate_fn=collate, batch_size=batch_size, num_workers=0,
                              shuffle=False)

print('dataset loaded')

it_test = iter(loader_test)
for step, (past, future, past_rel, future_rel, length) in enumerate(tqdm.tqdm(it_test)):
    print('insert your model and training loop')
