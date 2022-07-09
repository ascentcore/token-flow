import json

output = {}
with open('output.txt') as f:
    text = f.read()
    term = None
    definition = None
    for line in text.splitlines():    
        if term is None:
            term = line
        elif definition is None:
            definition = line
        else:
            output[term] = definition
            term = None
            definition = None


json.dump(output, open('output.json', 'w'), indent=2)