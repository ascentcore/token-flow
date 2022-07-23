from net.context import Context
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

text = """
Plants and animals need a healthy environment to survive. An ecosystem is an area where living organisms interact in a specific way with the local environment to survive. When ecosystems are damaged by man, then some living organisms may not be able to survive. A biome is a large group of similar ecosystems like the desert, savanna, and rainforest.
Environmental science studies the environment and how the earth works. Environmental scientists often study how humans have impacted the Earth environment and how we can reduce the impact that humans have on the environment.
Environmental scientists study things like the atmosphere, the oceans, geology, habitats, and ecology.
The Earth environment is constantly recycling nutrients so they can be used by different parts of the environment. These cycles are important for the existence of living organisms. Some important cycles include the water cycle, the nitrogen cycle, the carbon cycle, the oxygen cycle, and the food chain.
Human activities have created many environmental issues from land, water, and air pollution. Part of environmental science is to determine how the environment has been impacted and then to work on ways to help the environment recover.
One important aspect of helping the environment to recover is renewable energy. Renewable energy uses energy sources that cannot be "used up." Rather than burning fossil fuels like coal and oil, renewable energy uses energy sources like the wind and the Sun.
Deserts are primarily defined by their lack of rain. They generally get 10 inches or less rain in a year. Deserts are characterized in an overall lack of water. They have dry soil, little to no surface water, and high evaporation. They are so dry that sometimes rain evaporates before it can hit the ground!
Hot in the Day, Cold at Night
Because deserts are so dry and their humidity is so low, they have no "blanket" to help insulate the ground. As a result, they may get very hot during the day with the sun beating down, but don't hold the heat overnight. Many deserts can quickly get cold once the sun sets. Some deserts can reach temperatures of well over 100 degrees F during the day and then drop below freezing (32 degrees F) during the night.
Where are the major hot and dry deserts?
"""


class CustomDS(Dataset):
    """
    This is a custom dataset class. It can get more complex than this, but simplified so you can understand what's happening here without
    getting bogged down by the preprocessing
    """

    def __init__(self, ds):
        self.X = [data.x for data in ds]
        self.Y = [data.y for data in ds]
        if len(self.X) != len(self.Y):
            raise Exception("The length of X does not match the length of Y")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # note that this isn't randomly selecting. It's a simple get a single item that represents an x and y
        _x = self.X[index]
        _y = self.Y[index]

        return _x, _y


class Residual(torch.nn.Module):
    def __init__(self, prev_features_count, mid_count, next_features_count):
        super(Residual, self).__init__()

        self.lin1 = nn.Linear(prev_features_count, mid_count)
        self.lin2 = nn.Linear(mid_count, next_features_count)

    def forward(self, data):
        x = data
        residual = x
        x = self.lin1(x)
        # x = F.dropout(x, p=0.5)
        x = F.relu(x)        
        x = self.lin2(x)
        # x = F.dropout(x, p=0.5)
        # x = F.relu(x)

        # x += residual
        x = F.relu(x)
        

        return x, residual


class Model(torch.nn.Module):
    def __init__(self, num_classes):
        super(Model, self).__init__()
        torch.manual_seed(12345)



        self.res1 = Residual(num_classes, int(num_classes*0.75), int(num_classes / 2))
        self.res2 = Residual(int(num_classes/2), int(num_classes*0.5), int(num_classes / 4))
        self.res3 = Residual(int(num_classes/4), int(num_classes*5), int(num_classes / 2))
        self.res4 = Residual(int(num_classes/2), int(num_classes*0.75), num_classes)

    def forward(self, data):
        x = data

        x, residual_1 = self.res1(x)
        x, residual_2 = self.res2(x)
        x, residual_3 = self.res3(x)
        x, residual_4 = self.res4(x)

        return x


def run_sample():
    print('Neural Net #2')

    context = Context()
    context.from_text(
        text,
        connect_all=True,
        set_graph=True)

    # context.render('./output/ds_sample2.png', consider_stimulus=False)
    # dataset = []
    # for token in [' <start> ']+text.lower().replace('\n', '').replace(',', ' ').replace('.', ' <end> <start> ').split():
    #     token = token.strip()
    #     print(token)
    #     data = context.get_tensor_from_nodes(
    #         context.G, (token, token), with_state=True)
       
    #     context.decrease_stimulus(context.G)
    #     context.stimulate_token(context.G, token)
    #     dataset.append(data)


    print(f'>>>>> Dataset length: {len(dataset)}')

    loader = DataLoader(CustomDS(dataset), batch_size=32, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Model(len(context.vocabulary))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    crit = torch.nn.CrossEntropyLoss()

    def train():
        model.train()

        loss_all = 0
        for batch_ndx, sample in enumerate(loader):
            x, y = sample
            data = x.to(device)
            optimizer.zero_grad()
            output = model(data)

            loss = crit(output, y.to(device))
            loss_all += loss

            loss.backward()
            optimizer.step()

        return loss_all


    print('Start training...')
    pbar = tqdm(range(1, 5000))
    for epoch in pbar:
        loss = train()
        pbar.set_description("Loss %s" % loss)


    context.decrease_stimulus(context.G, 1)
    token = None
    previous = None
    predict_index = None
    # for token in ['<start>', 'a', 'car']:
    for i in range(0, 10):
        context.decrease_stimulus(context.G)
        
        if previous != predict_index:
            previous = predict_index
            token = context.vocabulary[predict_index]
            context.stimulate_token(context.G, token)   

        data = context.get_tensor_from_nodes(context.G)
        print(data.x)

        output = model(data.x)
        predict_index = output.argmax()
        print(token, '->', context.vocabulary[predict_index])
        

        