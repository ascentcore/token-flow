import sys
import examples.context_sample1 as cs1
import examples.context_sample2 as cs2
import examples.context_sample3 as cs3

import examples.dataset_sample1 as ds1

samples = {
    'sample1': cs1,
    'sample2': cs2,
    'sample3': cs3,
    'sample4': ds1,
}

samples[sys.argv[1]].run_sample()
