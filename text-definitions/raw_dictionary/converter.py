import re

regex = re.compile("^[A-Z]+$")
with open('../dictionary/output.txt', 'w') as output:
    with open('pg29765.txt') as f:
        text = f.read()

        key = ''
        definition = ''
        can_digest = False
        for line in text.splitlines():
            if regex.match(line):
                key = key.strip()
                definition = definition.strip()
                if len(definition) > 10:
                    output.write(f'{key.lower()}\n{definition.lower()}\n\n')
                can_digest = False
                definition = ''
                key = line
            else :
                if not can_digest and 'Defn:' in line:
                    line = line[5:]
                    can_digest = True

                if can_digest:
                    definition += line
