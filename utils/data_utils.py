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

def generate_symbols(group_size, num_examples, base_alphabet='abcdefghijklmnopqrstuvwxyz'):
    """Generates a list of symbols dynamically for a given group size."""
    symbols = [''.join(random.choices(base_alphabet, k=random.randint(1, group_size))) for _ in range(num_examples)]
    return symbols

def generate_prompts_symbols(k, num_examples, group_range, context='random'):
    prompts = {}
    base_alphabet = 'abcdefghijklmnopqrstuvwxyz'
    
    # Generate a common set of context symbols from the base alphabet
    common_context_symbols = generate_symbols(max(group_range), k * num_examples, base_alphabet)
    
    for i in group_range:
        prompts[i] = []
        
        for j in range(k):
            # Select `num_examples` random symbols for context
            context_symbols = random.sample(common_context_symbols, num_examples)
            
            # Generate the last input symbol with exactly `i` letters
            last_input_symbol = ''.join(random.choices(base_alphabet, k=i))
            
            # Construct the full symbol list
            full_symbol_list = context_symbols + [last_input_symbol]
            
            # Format the prompt
            formatted_prompt = ','.join(f"{s}={s}" for s in full_symbol_list[:-1]) + f",{full_symbol_list[-1]}="
            prompts[i].append(formatted_prompt)
    
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
