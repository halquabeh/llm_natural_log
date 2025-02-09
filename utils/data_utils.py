def generate_number_prompts(number_list):
    number_list = ','.join([f"{num}={num}" for num in number_list[:-1]]) + f",{number_list[-1]}="
    return number_list

import random
# random.seed(42)
def generate_prompts_numerals(k, num_examples, upper_bound, group_range,interval_function , context='random'):
    numbs = {}
    for i in group_range:
        interval = interval_function(i)
        numbs[i] = random.choices(interval, k=k)

    prompts = {}
    for i, numb_list in numbs.items():
        prompts[i] = []
        for numb in numb_list:
            if context == 'random':
                numbers = [random.randint(0, upper_bound) for _ in range(num_examples)] + [numb]
            elif context == 'fixed':
                numbers = [6,2,8] + [numb]
            else:
                numbers = numbs[i][:num_examples] + [numb]
            prompts[i].append(generate_number_prompts(numbers))
    
    return prompts



def generate_symbol_prompts(symbol_list, group_size):
    # Join all symbols except the last one with commas
    context = ','.join([f"{sym}={sym}" for sym in symbol_list[:-1]])
    # The last input before = should have `group_size` symbols
    last_input = ''.join(symbol_list[-group_size:])
    return f"{context},{last_input}="

import itertools
# random.seed(42)

# Function to generate symbols dynamically for a given group size
def generate_symbols(group_size, num_examples):
    # Define the alphabet
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    # Generate all possible combinations of length `group_size`
    symbols = [''.join(comb) for comb in itertools.product(alphabet, repeat=group_size)]
    # Shuffle the symbols to ensure randomness
    random.shuffle(symbols)
    # If num_examples exceeds the number of possible symbols, reuse symbols
    if num_examples > len(symbols):
        symbols = random.choices(symbols, k=num_examples)
    else:
        symbols = symbols[:num_examples]
    return symbols

def generate_prompts_symbols(k, num_examples, group_range, context='random'):
    prompts = {}
    for i in group_range:
        # Generate symbols dynamically for the current group size
        interval = generate_symbols(i, k * (num_examples + 1))
        symbs = [interval[j * (num_examples + 1):(j + 1) * (num_examples + 1)] for j in range(k)]

        prompts[i] = []
        for symb_group in symbs:
            if context == 'random':
                # Randomly select symbols for the context
                context_symbols = symb_group[:num_examples]
            elif context == 'fixed':
                # Use a fixed set of symbols for the context
                context_symbols = ['A', 'B', 'C']
            else:
                # Use the first `num_examples` symbols from the interval
                context_symbols = interval[:num_examples]
            
            # The last input should have `i` symbols (group size)
            if len(symb_group) >= num_examples + 1:
                last_input_symbols = [symb_group[-1]]  # Use the last symbol in the group
                full_symbol_list = context_symbols + last_input_symbols
                prompts[i].append(generate_symbol_prompts(full_symbol_list, i))
            else:
                # Handle cases where there aren't enough symbols
                print(f"Warning: Not enough symbols for group {i}. Skipping this prompt.")
    
    return prompts

def string_to_number(s):
    """Converts a string to a base-26 number, where 'a' = 0, 'b' = 1, ..., 'z' = 25."""
    num = 0
    for char in s:
        num = num * 26 + (ord(char) - ord('a')) #get_numerical_value(char, mapping)#
    return num

def strings_to_numbers(strings):
    """Converts a list of strings to a list of base-26 numbers."""
    return [string_to_number(s) for s in strings]
