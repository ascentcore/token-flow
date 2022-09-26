create_dataset = True
save_model = True
folder = 'stories'
repeats = 5

initial_weight = 0.575
weight_increase = 0.073
temp_decrease = 0.127
neuron_opening = 0.75
decrease_on_end = 0.15

# Output vector will keep only the highest value
filtered_output = True

render_context = False

lr = 3e-4
epochs = 1000
batch_size = 64

pre = ""

generate_size = 20
prevent_convergence_history = None
