import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(12345)


class Trainer():

    def __init__(self, model, vocabulary):
        self.vocabulary = vocabulary
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)

        # self.loss_function = torch.nn.MSELoss()
        self.loss_function = torch.nn.CrossEntropyLoss()
        # self.loss_function = torch.nn.BCELoss()

        # self.optimizer = torch.optim.Adam(model.parameters(),
        #                                   lr=1e-1,
        #                                   weight_decay=1e-8)

        self.optimizer = torch.optim.Adam(model.parameters(),  lr=1e-4)
        # self.optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    def _inner_train(self, loader):
        self.model.train()

        loss_all = 0

        # pbar = tqdm(enumerate(loader))
        # for batch_ndx, sample in pbar:
        for batch_ndx, sample in enumerate(loader):
            x, y = sample
            data = x.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)

            loss = self.loss_function(output, y.to(self.device))
            loss_all += loss

            loss.backward()
            self.optimizer.step()

            # pbar.set_description(f"Loss {loss} / {batch_ndx}")

        return loss_all

    def get_sentence(self, context, input_data, generate_length=20, prevent_convergence_history=5, stimulus=None, num_patches=None):
        history = []
        sentence = ""

        for _ in range(0, generate_length):
            # x = torch.tensor(context.get_stimuli(), dtype=torch.float32)
            x = torch.tensor([input_data[-num_patches:]], dtype=torch.float32)
            output = self.model(x.to(self.device))
            predict_index = output[0].argmax()
            predict_value = self.vocabulary.vocabulary[predict_index]

            # top_keys = torch.topk(
            #     output[0], k=prevent_convergence_history + 1).indices.tolist()
            # top_keys = [x for x in top_keys if x not in history]

            # predict_index = top_keys[0]
            # predict_value = self.vocabulary.vocabulary[predict_index]

            # if (predict_value == '<start>' or predict_value == '<end>') and len(history) > 0:
            #     break

            history.append(predict_index)
            history = history[-prevent_convergence_history:]
            context.stimulate(predict_value, stimulus=stimulus)
            input_data.append(context.get_stimuli())
            sentence += predict_value + " "

        return sentence.strip()

    def generate(self, context, test_sentence="", generate_length=20, prevent_convergence_history=5):
        context.decrease_stimulus(1)
        context.stimulate_sequence(test_sentence)
        history = []
        sentence = ""

        for _ in range(0, generate_length):
            x = torch.tensor(context.get_stimuli(), dtype=torch.float32)
            output = self.model(x.to(self.device))
            predict_index = output.argmax()
            predict_value = self.vocabulary.vocabulary[predict_index]

            top_keys = torch.topk(
                output, k=prevent_convergence_history + 1).indices.tolist()
            top_keys = [x for x in top_keys if x not in history]

            predict_index = top_keys[0]
            predict_value = self.vocabulary.vocabulary[predict_index]
            # print(predict_value + ' > ',[self.vocabulary.vocabulary[pid] for pid in top_keys])

            history.append(predict_index)
            history = history[-prevent_convergence_history:]

            context.stimulate(predict_value)
            sentence += predict_value + " "

        print(test_sentence + ' > ' + sentence)

    def train(self, ds, epochs=10, batch_size=64):

        # self.model.train()
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

        pbar = tqdm(range(1, epochs))
        for epoch in pbar:
            # for epoch in range(1, epochs):
            loss = self._inner_train(loader)
            pbar.set_description(f"Loss {loss}")
