create_dataset = True
folder = 'stories'
repeats = 5

initial_weight = 0.375
weight_increase = 0.073
temp_decrease = 0.027
neuron_opening = 0.75
decrease_on_end = 0.15

# Output vector will keep only the highest value
filtered_output = True

render_context = False

lr = 1e-4
epochs = 100
batch_size = 64

pre = "No sooner were the ceremonies of the wedding over but the mother-in-law began to show herself in her true colors."

generate_size = 300
prevent_convergence_history = 5
