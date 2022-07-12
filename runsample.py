import sys
import examples.context_sample as cs1
import examples.context_full_sample as cs2
import examples.context_from_folder as cs3

samples = {
    'sample1': cs1,
    'sample2': cs2,
    'sample3': cs3
}

samples[sys.argv[1]].run_sample()
