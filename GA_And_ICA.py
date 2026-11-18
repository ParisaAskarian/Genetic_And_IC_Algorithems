import numpy as np

# the objective function must be minimum
def objective_function(x):
    return np.sum(np.square(x))

problem_dim = 5      # the number of problems
lower_bound = -10
upper_bound = 10
pop_size = 20
max_iter = 100

# --- (GA) ---
def genetic_algorithm():
    # make the population
    pop = np.random.uniform(lower_bound, upper_bound, (pop_size, problem_dim))
    
    for gen in range(max_iter):
        # caculate the score
        scores = np.array([objective_function(ind) for ind in pop])
        
        # choose the best ones in papulation
        best_indices = np.argsort(scores)[:pop_size//2]
        parents = pop[best_indices]
        
        # make the childrens( crossover )
        offspring = []
        for i in range(0, len(parents), 2):
            child = (parents[i] + parents[i+1]) / 2
            offspring.append(child + np.random.normal(0, 0.1, problem_dim)) # with a little jump
            
        pop = np.vstack((parents, offspring)) # replace the papulation
        
    best_score = np.min([objective_function(ind) for ind in pop])
    return best_score

# --- (ICA) ---
def imperialist_competitive():
    # make the countries
    countries = np.random.uniform(lower_bound, upper_bound, (pop_size, problem_dim))
    costs = np.array([objective_function(c) for c in countries])
    
    # seperation the imperials and colonies
    idx = np.argsort(costs)
    imperials = countries[idx[:2]]
    colonies = countries[idx[2:]]
    
    for it in range(max_iter):
        # Assimilation : colonies moving to imperials
        for i in range(len(colonies)):
            target_imp = imperials[0] if i < len(colonies)//2 else imperials[1]
            # moving slowlly
            colonies[i] += 0.2 * (target_imp - colonies[i]) 
            
        # caculate the new costs
        costs_col = np.array([objective_function(c) for c in colonies])
        costs_imp = np.array([objective_function(c) for c in imperials])
        
        # if a coloni is better than a imperial, replace it
        for i in range(len(colonies)):
            imp_idx = 0 if i < len(colonies)//2 else 1
            if costs_col[i] < costs_imp[imp_idx]:
                colonies[i], imperials[imp_idx] = imperials[imp_idx], colonies[i].copy()

    return np.min(costs_imp)

print(f"GA Best Result: {genetic_algorithm():.5f}")
print(f"ICA Best Result: {imperialist_competitive():.5f}")