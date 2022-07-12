from net.context import Context


def run_sample():
    print('Running sample 1')
    context = Context()
    context.G = context.from_text(
        #'A car is a vehicle that has wheels, carries a small number of passengers and is moved by an engine or a motor. Ford was the first mass production car. Initially, cars had steam engines.',
        'A machine is a device that does a physical task. Some machines make moving or lifting things easier. Other machines carry people from place to place. Yet other machines help in building or making things.',
        connect_all=False)

    context.render('./output/sample.jpg', consider_stimulus=False)
