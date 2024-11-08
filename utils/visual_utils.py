import matplotlib.pyplot as plt

def plot_predictions_across_layers(prompts, hidden_states):
    fig, ax = plt.subplots(len(hidden_states), 1, figsize=(10, 15))
    for i, state in enumerate(hidden_states):
        ax[i].imshow(state)
        ax[i].set_title(f"Layer {i} Hidden States")
    plt.show()
