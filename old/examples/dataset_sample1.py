from net.dataset import ContextualGraphDataset
import torch

def run_sample():
    dataset = ContextualGraphDataset(
        source='./datasets/simple', from_scratch=True)

    vocabulary = dataset.get_vocabulary_keys()


    for i in range (2, 3):
        data = dataset[i]
        index = 0
        # for item in data.x:
        #     print(vocabulary[index], item.item())
        #     index = index + 1
        print(vocabulary)
        print(i, vocabulary[torch.argmax(data.y).item()])
        print(data.x.squeeze())
        print(data.y)    
        print(data.edge_index)
        print(data.edge_attr)
        