import sys
import examples.context_sample1 as cs1
import examples.context_sample2 as cs2
import examples.context_sample3 as cs3
import examples.context_sample4 as cs4

import examples.dataset_sample1 as ds1

samples = {
    'cs1': cs1,
    'cs2': cs2,
    'cs3': cs3,
    'cs4': cs4,
    'ds1': ds1,
}

samples[sys.argv[1]].run_sample()
