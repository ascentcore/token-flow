import re
import os
import json

pattern = re.compile("^\d ")

kids_dict = open('assets/kids_dictionary.txt', 'r')

keyword = None
keyword_type = None
variations = None

keywords = {}
definitions = []


def process_def(line):
    line = re.sub(' \(see \w+ on page \d+\)', '', line)
    line = re.sub('^\d ', '', line)

    return line


for line in kids_dict.readlines():

    line = line.strip().lower()

    if line == '':
        if keyword is not None:
            keywords[keyword] = {'type': keyword_type,
                                 'variations': variations,
                                 'definitions': definitions}

        keyword = None
        variations = None
        keyword_type = None
        definitions = []
        continue
    elif keyword is None:
        keyword = line

        try:
            line.index(' ')
            print(line)
        except:
            pass

    elif keyword != None:
        if keyword_type == None:
            # keyword_type = line
            type_split = line.split(' ')
            keyword_type = type_split[-1]
            variations = type_split[:-1]
        elif len(definitions) == 0 or pattern.match(line):

            definitions.append(process_def(line))


with open('assets/kids_dictionary.json', 'w') as outfile:
    outfile.write(json.dumps(keywords, indent=2))
