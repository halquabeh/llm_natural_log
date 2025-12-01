# readme Natural Log

# Start with installing the dependencies from requirements.txt
> pip install -r requirements.txt # make sure you are in the right env, if not then create one beforehand.

# the main file
# main_lab.py is the class containter that has analyzer, which is class that runs all prompts in the list of groups by the args.model and collect the state from each layer, further the results are transformeed to low space using PCA or PLS and saved to folder ICLR_results