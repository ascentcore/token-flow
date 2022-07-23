from net.context import Context


def run_sample():
    print('Running sample 1')

    text = """
    A car is a vehicle that has wheels. The wheel is round.
    """

    context = Context()
    context.from_text(
        text,
        connect_all=True,
        set_graph=True)
    print(context.vocabulary)
    context.stimulate_token(context.G, '<start>')
    context.decrease_stimulus(context.G, .3)
    context.stimulate_token(context.G, 'car')
    context.decrease_stimulus(context.G, .3)
    context.stimulate_token(context.G, 'vehicle')
    context.render('./output/sample.jpg', title=text,
                   consider_stimulus=True, arrowsize=5)

# A machine is a device that does a physical task. Some machines make moving or lifting things easier. Other machines carry people from place to place. Yet other machines help in building or making things.
