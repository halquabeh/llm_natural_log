def generate_number_prompts(number_list):
    number_list = ','.join([f"{num}={num}" for num in number_list[:-1]]) + f",{number_list[-1]}="
    return number_list

import random
random.seed(42)
def generate_prompts(k, num_examples, upper_bound, group_range,interval_function , context='random'):
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