# Number Representations in LLMs: A Computational Parallel to Human Perception
![Figure 0: Description of Figure](overview.png)

## Table of Contents
- [Motivation](#motivation)
- [Results](#results)
- [Figures](#figures)
- [Citations](#citations)
- [How to Run the Code](#how-to-run-the-code)
- [Contributors](#contributors)
- [License](#license)

## Motivation
Humans are believed to perceive numbers on a logarithmic mental number line, where smaller values are represented with greater resolution than larger ones. Inspired by this hypothesis, we investigate whether large language models (LLMs) exhibit a similar logarithmic-like structure in their internal numerical representations. By analyzing how numerical values are encoded across different layers of LLMs, we apply dimensionality reduction techniques such as PCA and PLS followed by geometric regression to uncover latent structures in the learned embeddings. Our findings reveal that the model’s numerical representations exhibit sublinear spacing, with distances between values aligning with a logarithmic scale. This suggests that LLMs, much like humans, may encode numbers in a compressed, non-uniform manner.

## Results

Our approach achieves the following key results:

- **Result 1**: [Briefly describe the result, e.g., "Improved accuracy by X% compared to baseline."]
- **Result 2**: [e.g., "Reduced computation time by Y%."]
- **Result 3**: [e.g., "Demonstrated robustness under Z conditions."]

Below are some visualizations of our results:

![Figure 1: Description of Figure](plot1.png)
*Caption: A brief explanation of the figure.*

![Figure 2: Description of Figure](plot2.png)
*Caption: Another brief explanation of the figure.*



## Citations

If you find our work useful, please cite our paper:

```bibtex
@article{your_paper_key,
  title={Your Paper Title},
  author={Author1, Firstname and Author2, Firstname},
  journal={Journal/Conference Name},
  year={Year},
  volume={Volume},
  number={Issue},
  pages={Pages}
}

## How to Run the Code**
```markdown

To reproduce our results, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/your_repo_name.git
   cd your_repo_name

2- python main.py

3- To reporoduce the results in the paper, run the following code.


