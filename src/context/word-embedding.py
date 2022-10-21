import numpy as np
import math

n_dim = 2
characters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 
              'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
               'w', 'x', 'y', 'z']
total = len(characters)
dim_len = math.ceil(math.pow(total, 1 / n_dim))
print('Dimension size:', dim_len)

def getForIndex(index):
  dim_arr = []
  rest = index
  for i in reversed(range(0, n_dim)):
    dim = rest / math.pow(dim_len, i);
    rest = rest % math.pow(dim_len, i);

    dim_arr.append(math.floor(dim))

  return dim_arr

def getLocation(word):
  letters = list(word)
  letter_locations = [getForIndex(characters.index(letter)) for letter in letters]
  print(letter_locations)
  weighted_letter_locations = [np.array(location) * 10**(-i) for i, location in enumerate(letter_locations)]
  print(weighted_letter_locations)
  word_location = [sum(x) for x in zip(*weighted_letter_locations)]

  return word_location
      
word_location = getLocation('apple')
print(word_location)

