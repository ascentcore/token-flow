import random
from fastapi import HTTPException, FastAPI, Response, Depends, Request
from uuid import UUID, uuid4
from fastapi.middleware.cors import CORSMiddleware

from starlette.middleware.sessions import SessionMiddleware
from src.context.dataset import Dataset
from src.context.vocabulary import Vocabulary
from src.context.context import Context

import networkx as nx

vocabulary = Vocabulary(
    accept_all=True,
    include_start_end=False,
    include_punctuation=False,
    use_lemma=True,
    add_lemma_to_vocab=True)

dataset = Dataset(vocabulary)


dataset.add_context(Context('Environment', vocabulary,
                            initial_weight=0.1,
                            weight_increase=0.1,
                            neuron_opening=0.95,
                            temp_decrease=0.05))

dataset.add_context(Context('Pollution', vocabulary,
                            initial_weight=0.1,
                            weight_increase=0.1,
                            neuron_opening=0.95,
                            temp_decrease=0.05))

dataset.delete_context('default')


# context2 = Context('context2', vocabulary,
#                   initial_weight=0.2,
#                   neuron_opening=0.95,
#                   weight_increase=0.1,
#                   temp_decrease=0.05)

# dataset.add_context(context1)
# dataset.add_context(context2)

# dataset.get_context('context1').add_text(
#     'The rain in spain falls mainly on the plain')
# dataset.get_context('context2').add_text(
#     'Once upon a time there was a little red riding hood')

# dataset.store('contexts/test')

# dataset = Dataset.load('contexts/test')


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/contexts')
def read_root(request: Request):
    return [key for key in dataset.contexts.keys()]


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


@app.post('/stimulate')
async def read_root(request: Request):
    body = await request.json()
    for context in dataset.contexts:
        dataset.get_context(context).stimulate_sequence(body['text'])

    return random.randint(1, 100)


@app.post('/reset_stimuli')
async def reset_stimuli():
    for context in dataset.contexts:
        dataset.get_context(context).decrease_stimulus(1)

    return random.randint(1, 100)


@app.get('/context/{name}/graph')
def read_root(name):
    G = dataset.get_context(name).graph
    d = nx.json_graph.node_link_data(G)

    return d
