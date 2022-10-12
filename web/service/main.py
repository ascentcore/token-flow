import random
from fastapi import HTTPException, FastAPI, Response, Depends, Request
from uuid import UUID, uuid4
from fastapi.middleware.cors import CORSMiddleware

from starlette.middleware.sessions import SessionMiddleware
from src.context.dataset import Dataset
from src.context.vocabulary import Vocabulary
from src.context.context import Context

import networkx as nx

# dataset = Dataset.load('/app/studies/chat/dataset')
# dataset.delete_context('default')


# vocabulary = Vocabulary(include_start_end=False,
#                         include_punctuation=False,
#                         accept_all=False,
#                         use_token=False)

# context1 = Context('context1', vocabulary,
#                    initial_weight=0.2,
#                    neuron_opening=0.95,
#                    weight_increase=0.1,
#                    temp_decrease=0.05)

# context2 = Context('context2', vocabulary,
#                    initial_weight=0.2,
#                    neuron_opening=0.95,
#                    weight_increase=0.1,
#                    temp_decrease=0.05)

# dataset = Dataset(vocabulary=vocabulary)
# dataset.add_context(context1)
# dataset.add_context(context2)
# dataset.delete_context('default')

# dataset.get_context('context1').add_text(
#     'The rain in spain falls mainly on the plain')
# dataset.get_context('context2').add_text(
#     'Once upon a time there was a little red riding hood')

# dataset.store('datasets/basic')

# dataset.store('contexts/test')

datasets = [
    Dataset(vocabulary=Vocabulary(include_start_end=False,
                                  include_punctuation=False,
                                  accept_all=True,
                                  use_token=False), name="Basic Dataset"),
    Dataset.load('contexts/test')
]

dataset = datasets[0]


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/contexts')
def get_contexts(request: Request):
    return [key for key in dataset.contexts.keys()]


@app.get('/datasets')
def get_datasets(request: Request):
    return [dataset.settings['name'] for dataset in datasets]

@app.post('/switch/{id}')
async def read_root(request: Request, id):
    global dataset
    dataset = datasets[int(id)]

@app.get('/vocabulary')
def read_vocabulary():
    return dataset.vocabulary.vocabulary


@app.post('/add_text')
async def read_root(request: Request):
    body = await request.json()
    dataset.add_text(body['text'])

    return random.randint(1, 100)


@app.post('/add_text/{context}')
async def read_root(request: Request, context):
    body = await request.json()
    dataset.get_context(context).add_text(body['text'])
    return random.randint(1, 100)


@app.post('/create_context')
async def read_root(request: Request):
    body = await request.json()

    context = Context(body['name'], dataset.vocabulary,
                      initial_weight=body['initial_weight'],
                      weight_increase=body['weight_increase'],
                      temp_decrease=body['temp_decrease'])

    dataset.add_context(context)


@app.post('/stimulate')
async def read_root(request: Request, context=None):
    body = await request.json()
    text = body['text']
    for context in dataset.contexts if context == None else [dataset.get_context(context)]:
        if " " in text:
            dataset.get_context(context).stimulate_sequence(text)
        else:
            dataset.get_context(context).stimulate(text)

    return random.randint(1, 100)


@app.post('/stimulate/{context}')
async def read_root(request: Request, context):
    body = await request.json()
    text = body['text']
    if " " in text:
        dataset.get_context(context).stimulate_sequence(text)
    else:
        dataset.get_context(context).stimulate(text)

    return random.randint(1, 100)


@app.post('/reset_stimuli')
async def reset_stimuli():
    for context in dataset.contexts:
        dataset.get_context(context).decrease_stimulus(1)

    return random.randint(1, 100)


@app.post('/reset_stimuli/{context}')
async def reset_stimuli(context):
    dataset.get_context(context).decrease_stimulus(1)

    return random.randint(1, 100)


@app.get('/context/{name}/graph')
def read_root(name):
    G = dataset.get_context(name).graph
    d = nx.json_graph.node_link_data(G)

    return d
