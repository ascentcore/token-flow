from net.context import Context


def run_sample():
    print('Parsing folder text')
    context = Context()

    context.from_folder('./datasets/simplest', reset=True, connect_all=False)

    print(context.vocabulary)
