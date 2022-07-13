from net.context import Context
import imageio
import spacy
nlp = spacy.load("en_core_web_sm")


def run_sample():
    print('Running sample 1')
    context = Context()
    with open('examples/assets/environment.txt', 'r') as f:
        text = f.read()
        G = context.from_text(text, connect_all=True)

        images = []
        doc = nlp("""
        Climate change is already affecting water access for people around the world, causing more severe droughts and floods. Increasing global temperatures are one of the main contributors to this problem. Climate change impacts the water cycle by influencing when, where, and how much precipitation falls.
            """)
        pos = None
        prev = []
        for token in doc:
            lower = token.lemma_.lower()
            if lower in context.vocabulary:
                index = context.vocabulary.index(lower)
                if index in G.nodes():
                    print(f'stimulating {token.lemma_.lower()}')
                    context.stimulate(G, context.vocabulary.index(
                        token.lemma_.lower()), 1, and_set=True)

            prev.append(token.text)
            pos = context.render('./output/sample.jpg',
                                 ' '.join(prev), pre_pos=pos, arrowsize=1)
            images.append(imageio.imread('./output/sample.jpg'))
            context.decrease_stimulus(G, 0.02)

            prev = prev[-8:]

        for i in range(0, 20):
            context.render('./output/sample.jpg', ' '.join(prev), arrowsize=1)
            images.append(imageio.imread('./output/sample.jpg'))
            context.decrease_stimulus(G, 0.1)

        imageio.mimsave('output/flow.gif', images)
