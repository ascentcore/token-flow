from net.context import Context


def run_sample():
    print('Running sample 1')
    context = Context()
    context.G = context.from_text(
        'A car is a vehicle that has wheels, carries a small number of passengers, and is moved by an engine or a motor. Ford was the first mass production car. Initially, cars had steam engines.',
        connect_all=False)

    for to_stimulate in ['<start>', 'a', 'car', 'has', 'wheels']:
        lemma = context.get_lemma(to_stimulate)
        print('stimulating', to_stimulate, lemma)
        context.stimulate_token(context.G, lemma)
        context.render('./output/sample.jpg', consider_stimulus=True)
