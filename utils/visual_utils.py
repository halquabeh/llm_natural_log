import matplotlib.pyplot as plt
import numpy as np
def plot_predictions_across_layers(prompts, hidden_states):
    fig, ax = plt.subplots(len(hidden_states), 1, figsize=(10, 15))
    for i, state in enumerate(hidden_states):
        ax[i].imshow(state)
        ax[i].set_title(f"Layer {i} Hidden States")
    plt.show()
import os
import numpy as np
import matplotlib.pyplot as plt



import os
import numpy as np
import matplotlib.pyplot as plt

def plot_pca_projections(final_results, file_path):
    """
    Plots PCA projections for each layer, coloring points by their group ID, and saves the figure.
    
    Args:
        final_results (dict): Dictionary containing layers as keys and transformed hidden states + answers as values.
        file_path (str): Path where the figure should be saved.
    """
    num_layers = len(final_results)
    fig, axs = plt.subplots(nrows=num_layers, ncols=1, figsize=(6, 3 * num_layers), constrained_layout=True)
    
    if num_layers == 1:
        axs = [axs]  # Ensure axs is iterable when there's only one layer
    
    for idx, (layer, data) in enumerate(final_results.items()):
        all_answers = []
        projections = []
        new_array = []
        
        for group_id, group_data in data['answers'].items():
            all_answers.extend(group_data)
            projections.extend(data['hidden_states'][group_id])
            new_array.extend([int(group_id)] * len(group_data))
        
        all_answers = np.array(all_answers, dtype=float)
        projections = np.array(projections).squeeze()
        
        axs[idx].scatter(np.log10(all_answers), projections, c=new_array, cmap='viridis', alpha=0.7)
        axs[idx].set_title(f'Layer {layer}')
        axs[idx].set_xlabel('$\log_{10}$(original values)')
        axs[idx].set_ylabel('Transformed Hidden State Projection')
    
    plt.savefig(file_path + '.png', dpi=300)
    plt.close()
    print(f'Figure saved at')


