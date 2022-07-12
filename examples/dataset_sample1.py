from net.dataset import ContextualGraphDataset


def run_sample():
    dataset = ContextualGraphDataset(
        source='./datasets/simplest', prune_vocabulary=True, from_scratch=True)
