import random
from fastapi import HTTPException, FastAPI, Response, Depends, Request
from uuid import UUID, uuid4
from fastapi.middleware.cors import CORSMiddleware

from starlette.middleware.sessions import SessionMiddleware
from src.context.dataset import Dataset
from src.context.vocabulary import Vocabulary
from src.context.context import Context

import networkx as nx

vocabulary = Vocabulary(accept_all=True,
                        include_start_end=False,
                        include_punctuation=False,
                        use_lemma=False,
                        add_lemma_to_vocab=False)


for token in 'the rain in spain falls mainly on the plain the rain helps plants to grow'.split():
    vocabulary.add_to_vocabulary(token)


sample1 = Dataset(Vocabulary(accept_all=True,
                             include_start_end=False,
                             include_punctuation=False,
                             use_lemma=False,
                             add_lemma_to_vocab=False), name='Engine')

mechanical = Context('mechanical', sample1.vocabulary,
                     initial_weight=0.5,
                     weight_increase=0.1,
                     temp_decrease=0.05)

mechanical.add_text('''The engine of a vehicle is the part that converts the energy of the fuel into mechanical energy, and produces the power which makes the vehicle move. He got into the driving seat and started the engine. Automotive Engines are generally classified according to following different categories: Internal combustion (IC) and External Combustion (EC) Type of fuel: Petrol, Diesel, Gas, Bio / Alternative Fuels. Number of strokes – Two stroke Petrol, Two-Stroke Diesel, Four Stroke Petrol / Four Stroke Diesel. The engine is the vehicle's main source of power. The engine uses fuel and burns it to produce mechanical power. The heat produced by the combustion is used to create pressure which is then used to drive a mechanical device. The engine is a lot like the brain of a car. It holds all the power necessary to help your car function. And without it, your car would be nothing. But there are multiple car engine types out there on the road. An engine, or motor, is a machine used to change energy into a movement that can be used. The energy can be in any form. Common forms of energy used in engines are electricity, chemical (such as petrol or diesel), or heat. When a chemical is used to produce the energy it is known as fuel.''')

software = Context('software', sample1.vocabulary,
                   initial_weight=0.5,
                   weight_increase=0.1,
                   temp_decrease=0.05)

software.add_text('''A software engine is a computer program, or part of a computer program, that serves as the core foundation for a larger piece of software. This term is often used in game development, in which it typically refers to either a graphics engine or a game engine around which the rest of a video game is developed. While the term can also be used in other areas of software development, its particular meaning can be more nebulous in those instances. A software engine can be developed by a company that is using it, or may be developed by another company and then licensed to other developers. When used in the general context of computer software development, a software engine typically refers to the core elements of a particular program. This usually does not include features such as the user interface (UI) and numerous art assets added to the core engine itself. For an operating system (OS), for example, the software engine might be the source code that establishes file hierarchy, input and output methods, and how the OS communicates with other software and hardware. The exact contents of such an engine can vary from program to program, however. In computer and console game development, a software engine typically refers to either a game’s graphics engine or the overall game engine. The graphics engine for a game is typically the software used to properly render out the graphics seen by players. This often uses art assets created in other programs, which are then ported into the graphics engine for use during game play. The use of a software engine for the graphics of a game can make rendering much easier, and may also simplify the process of ensuring software and hardware compatibility.''')

sample1.add_context(mechanical)
sample1.add_context(software)
sample1.delete_context('default')

sample2 = Dataset(Vocabulary(accept_all=True,
                             include_start_end=False,
                             include_punctuation=False,
                             use_lemma=False,
                             add_lemma_to_vocab=False), name='Politics')


macron = Context('Macron', sample2.vocabulary,
                 initial_weight=0.7,
                 weight_increase=0.1,
                 temp_decrease=0.05)

macron.add_text('''French President Emmanuel Macron said on Tuesday his phone calls with U.S. President Donald Trump were just like sausages: better not explain what’s inside.

During a news conference with Israeli Prime Minister Benjamin Netanyahu, Macron was asked about a CNN report on Monday saying that a phone call between the French leader and Trump last week had been “terrible”.

Borrowing a famous quote from 19th century Prussian statesman Otto von Bismarck, Macron summed up his policy of refraining from making off-the-record comments about his conversations with other world leaders.

“As Bismarck used to say, if we explained to people how sausages were made, it’s unlikely they’d keep eating them,” Macron told reporters.

“So I like it when people see the finished meal, but I’m not convinced the kitchen commentary helps with delivering the meal or eating it,” Macron added.

A readout from the French presidency of Macron’s call with Trump last week said the 40-year old president had told Trump that his decision to slap tariffs on the exports of U.S. allies were both “illegal” and a “mistake”.

The shorter White House readout of the same call said the conversation had focused on trade and immigration, without elaborating.

“Macron thought he would be able to speak his mind, based on the relationship. But Trump can’t handle being criticised like that,” CNN quoted an unidentified source as saying. “Just bad. It was terrible.”

Macron, appalled by his predecessor Francois Hollande’s frequent off-the-record comments to journalists, has kept the press at a distance since his election last year and banned his aides from giving behind-the-scenes accounts of his presidency.

“You can go and ask the people who make comments, but here in Paris we don’t make comments on how it went, or how hot, cold, warm or terrible things are. We just go ahead and do things,” Macron said.

He added that he would have a “useful” and “frank” exchange with Trump at the next G7 summit in Canada on June 7-8 about both issues on which they agree and those where they disagree.''')

sample2.delete_context('default')
sample2.add_context(macron)


datasets = [
    sample2,
    sample1,
    Dataset(vocabulary=vocabulary, name="Basic Dataset"),
    Dataset.load('contexts/agents'),
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


@ app.get('/contexts')
def get_contexts(request: Request):
    return [key for key in dataset.contexts.keys()]


@ app.get('/datasets')
def get_datasets(request: Request):
    return [dataset.settings['name'] for dataset in datasets]


@ app.post('/switch/{id}')
async def read_root(request: Request, id):
    global dataset
    dataset = datasets[int(id)]


@ app.get('/vocabulary')
def read_vocabulary():
    return dataset.vocabulary.vocabulary


@ app.post('/add_text')
async def read_root(request: Request):
    body = await request.json()
    dataset.add_text(body['text'])

    return random.randint(1, 100)


@ app.post('/add_text/{context}')
async def read_root(request: Request, context):
    body = await request.json()
    dataset.get_context(context).add_text(body['text'])
    print(context)
    print(body['text'])
    print(dataset.get_context(context).graph)
    return random.randint(1, 100)


@ app.post('/create_context')
async def read_root(request: Request):
    body = await request.json()

    context = Context(body['name'], dataset.vocabulary,
                      initial_weight=body['initial_weight'],
                      weight_increase=body['weight_increase'],
                      temp_decrease=body['temp_decrease'])

    dataset.add_context(context)


@ app.post('/stimulate')
async def read_root(request: Request, context=None):
    body = await request.json()
    text = body['text']
    for context in dataset.contexts if context == None else [dataset.get_context(context)]:
        if " " in text:
            dataset.get_context(context).stimulate_sequence(text)
        else:
            dataset.get_context(context).stimulate(text)

    return random.randint(1, 100)


@ app.post('/stimulate/{context}')
async def read_root(request: Request, context):
    body = await request.json()
    text = body['text']
    if " " in text:
        dataset.get_context(context).stimulate_sequence(text)
    else:
        dataset.get_context(context).stimulate(text)

    return random.randint(1, 100)


@ app.post('/reset_stimuli')
async def reset_stimuli():
    for context in dataset.contexts:
        dataset.get_context(context).decrease_stimulus(1)

    return random.randint(1, 100)


@ app.post('/reset_stimuli/{context}')
async def reset_stimuli(context):
    dataset.get_context(context).decrease_stimulus(1)

    return random.randint(1, 100)


@ app.get('/context/{name}/graph')
def read_root(name):
    G = dataset.get_context(name).graph
    d = nx.json_graph.node_link_data(G)

    return d
