import os
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(12345)


class Trainer():

    

    def __init__(self, model, vocabulary, config, lr=1e-3):
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

        # self.optimizer = torch.optim.Adam(model.parameters(),  lr=lr)
        # self.optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        self.grad_norm_clip = config.grad_norm_clip
        self.optimizer = model.configure_optimizers(config)

    def batch_train(self, sample):
        x, y = sample

        x_indices = []
        x_stimulus = []

        X = []
        for lst in x:
            X.append([list(e) for e in zip(lst[0], lst[1])])
        
        b_s = len(X[0])

        X_s = []
        for i in range(0, b_s):
            X_s.append([lst[i] for lst in X])
        
        for input in X_s:
            indices = []
            stimulus = []
            indices = [int(i) for i, _ in input]
            stimulus = [float(s) for _, s in input]

            x_indices.append(indices)
            x_stimulus.append(stimulus)

        x_indices = torch.tensor(x_indices, dtype=torch.int32).to(self.device)
        x_stimulus = torch.tensor(x_stimulus, dtype=torch.float32).to(self.device)

        for (x_i, y_j) in zip(x_indices, y):
            x_words = [self.vocabulary.vocabulary[i] for i in x_i]   
            y_target = self.vocabulary.vocabulary[(y_j == 1).nonzero(as_tuple=True)[0]]

            if os.path.exists(f'studies/single-context/input_report/report.log'):
                input_report_file = open(f'studies/single-context/input_report/report.log', "a")
                input_report_file.write(f'history {x_words[0:20]}\n')
                input_report_file.write(f'current {x_words[20]}\n')
                input_report_file.write(f'next {x_words[21:41]}\n')
                # input_report_file.write(f'history {x_words[0:50]}\n')
                # input_report_file.write(f'current {x_words[50]}\n')
                # input_report_file.write(f'next {x_words[51:81]}\n')
                input_report_file.write(f'target {y_target}\n')
                input_report_file.write('-----------------------------------------------------\n\n')

        
        logits, loss, acc = self.model(x_indices, x_stimulus, y.to(self.device))

        # self.optimizer.zero_grad()
        self.model.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)
        self.optimizer.step()

        return loss, acc

    def _inner_train(self, loader):
        self.model.train()

        loss_all = 0

        # pbar = tqdm(enumerate(loader))
        # for batch_ndx, sample in pbar:
        for batch_ndx, sample in enumerate(loader):
            x, y = sample
            # print([self.vocabulary.closest(x[:-1]) for x in x])
            data = x.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)

            loss = self.loss_function(output, y.to(self.device))
            loss_all += loss

            loss.backward()
            self.optimizer.step()

            # pbar.set_description(f"Loss {loss} / {batch_ndx}")

        return loss_all

    def predict(self, input_data):
        self.model.eval()
        # Previous version used a float type input
        # x = torch.tensor(input_data, dtype=torch.int32)
        x = torch.tensor(input_data, dtype=torch.int32)
        return self.model(x.to(self.device))
        # predict_index = int(output)
        # predict_value = self.vocabulary.vocabulary[predict_index]
        # return predict_index, predict_value

    def get_sentence(self, context, input_data, generate_length=20, prevent_convergence_history=5, stimulus=None, num_patches=None, break_on_end=False, break_on_eol=False):
        history = []
        sentence = ""

        for _ in range(0, generate_length):
            if num_patches is not None:
                x = torch.tensor([input_data[-num_patches:]],
                                 dtype=torch.float32)
            else:
                x = torch.tensor(context.get_stimuli(), dtype=torch.float32)

            output = self.model(x.to(self.device))
            if num_patches is not None:
                predict_index = output[0].argmax()
            else:
                predict_index = output.argmax()

            predict_value = self.vocabulary.vocabulary[predict_index]

            if prevent_convergence_history != None:
                if num_patches is not None:
                    top_keys = torch.topk(
                        output[0], k=prevent_convergence_history + 1).indices.tolist()
                else:
                    top_keys = torch.topk(
                        output, k=prevent_convergence_history + 1).indices.tolist()

                top_keys = [x for x in top_keys if x not in history]

                predict_index = top_keys[0]
                predict_value = self.vocabulary.vocabulary[predict_index]

            history.append(predict_index)
            if prevent_convergence_history != None:
                history = history[-prevent_convergence_history:]

            context.stimulate(predict_value, stimulus=stimulus)
            input_data.append(context.get_stimuli())
            sentence += predict_value + " "

            if (break_on_end == True and predict_value == '<end>') and len(history) > 0:
                break

            if (break_on_eol == True and predict_value == '<eol>'):
                break

        return sentence.strip()

    def generate(self, context, test_sentence="", generate_length=20, prevent_convergence_history=5):
        context.decrease_stimulus(1)
        context.stimulate_sequence(test_sentence)
        history = []
        sentence = ""
        last_token = None
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
            if last_token == predict_value:
                context.decrease_stimulus()
            else:
                context.stimulate(predict_value)
            sentence += predict_value + " "
            last_token = predict_value

        print(test_sentence + ' > ' + sentence)

    def train(self, ds, epochs=10, batch_size=64):

        # self.model.train()
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

        pbar = tqdm(range(1, epochs))
        for epoch in pbar:
            # for epoch in range(1, epochs):
            loss = self._inner_train(loader)
            pbar.set_description(f"Loss {loss}")
