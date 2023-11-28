#########################
##### Specify the model...

###############
######### States

using StatsBase

function generate_all_states(N_Quanta, N_Objects)

    all_states = []

    for x in Iterators.product(Iterators.repeated(0:N_Quanta, N_Objects)...)
        if sum(x) == N_Quanta
            push!(all_states, x)
        end
    end
    return all_states
end

function get_state_index(this_state,S)
    
    return findfirst(isequal(this_state), S)

end

######################
#  Transitions
function get_s_prime_given_action_and_degraded_object(s,A_object,degraded_object)
    
    s_prime = copy(s)
    s_prime[degraded_object]-=1
    s_prime[A_object]+=1
   
    return s_prime
    
end

function get_possible_s_prime_and_probs(s,A)
    
    N_Objects = length(s)
    
    N_Quanta = sum(s)
    
    # possible next states
    possible_S_prime = [get_s_prime_given_action_and_degraded_object(s,A,i) for i in 1:N_Objects]

    # probability that object is selected for degradation...
    prob_S_prime = s ./ N_Quanta#[s[i]/N_Quanta for i in 1:N_Objects]
    
    possible_S_prime = possible_S_prime[prob_S_prime .> 0]
    prob_S_prime = prob_S_prime[prob_S_prime .> 0]
    
    return possible_S_prime, prob_S_prime
end


########################
## Rewards
"""
function prob_remember(num_quanta)
    dv = 2*(-2.5 .+ num_quanta)
    return 1 ./ (1 .+exp.(-dv))
end
"""

#function prob_remember(num_quanta)
#    return .5 .+ ((.1*num_quanta).^.5)/2
#end
"""
function prob_remember(num_quanta)
    return 1 .- exp.(-.2*num_quanta)
end
"""

function prob_remember(num_quanta)
    return .5 .+ (1 .- exp.(.1*-(num_quanta)))./2
end

function get_state_reward(s, object_probe_probs, per_timestep_probe_prob)

    return prob_remember.(s)'*object_probe_probs*per_timestep_probe_prob

end

################################
## Value iteration

function get_optimal_V(S,object_probe_probs, per_timestep_probe_prob)
    
    NS = length(S)
    
    max_change = 1e3
    gamma = .99

    # initialize V
    V = zeros(NS)

    while max_change > 1e-9

        V_new = copy(V)

        # loop through each state
        for s_idx in 1:NS

            s = S[s_idx]

            legal_actions = findall(s.>0)

            N_legal_actions = length(legal_actions)

            # compute Q value for each action and take max
            Q = zeros(N_legal_actions)
            for a_idx in 1:N_legal_actions

                A = legal_actions[a_idx]

                # takes in list, returns list
                possible_S_prime, prob_S_prime = get_possible_s_prime_and_probs(collect(s),A)

                possible_R_s = [get_state_reward(this_s, object_probe_probs, per_timestep_probe_prob) for this_s in possible_S_prime]

                # takes in tuple
                possible_S_prime_idxs = [get_state_index(tuple(this_s...),S) for this_s in possible_S_prime]

                possible_V_s_prime = V[possible_S_prime_idxs]

                Q[a_idx] = prob_S_prime'*(possible_R_s + gamma*possible_V_s_prime)
            end

            V_new[s_idx] = maximum(Q)

        end

        max_change = maximum(abs.(V_new - V))
        V = V_new

    end
    
    return V
    
end

function get_optimal_policy(V, S,object_probe_probs, per_timestep_probe_prob)

    NS = length(S)
    N_Objects = length(S[1])
    policy = zeros(NS,N_Objects) # change this so that it's a probability distribution
    
    gamma = .95

    
    for s_idx in 1:NS

        # loop through each state

        s = S[s_idx]

        legal_actions = findall(s.>0)

        N_legal_actions = length(legal_actions)

        # compute Q value for each action and take max
        Q = zeros(N_legal_actions)
        for a_idx in 1:N_legal_actions

            A = legal_actions[a_idx]

            possible_S_prime, prob_S_prime = get_possible_s_prime_and_probs(collect(s),A)

            possible_R_s = [get_state_reward(this_s, object_probe_probs, per_timestep_probe_prob) for this_s in possible_S_prime]

            possible_S_prime_idxs = [get_state_index(tuple(this_s...),S) for this_s in possible_S_prime]

            possible_V_s_prime = V[possible_S_prime_idxs]

            Q[a_idx] = prob_S_prime'*(possible_R_s + gamma*possible_V_s_prime)
        end
        
        max_val = maximum(Q)
        
        max_Q_idxs = findall(Q .== max_val)
        max_actions = legal_actions[max_Q_idxs]
        
        policy[s_idx, max_actions] .= 1 ./ length(max_actions)        

    end
   
    return policy
    
end

function value_iteration(S,object_probe_probs, per_timestep_probe_prob)
    
    V = get_optimal_V(S,object_probe_probs, per_timestep_probe_prob)
    policy = get_optimal_policy(V, S,object_probe_probs, per_timestep_probe_prob)
    
    return policy, V
    
end

function get_random_quanta_policy(S)

    # quanta is randomly chosen...
    
    NS = length(S)
    N_Objects = length(S[1])

    random_policy = zeros(NS,N_Objects)

    # each quanta equally likely to be selected
    for s_idx in 1:NS
        s = collect(S[s_idx])
        random_policy[s_idx,:] = s / sum(s)
    end
    
    return random_policy
    
end

function get_epsilon_policy(optimal_policy, random_policy, epsilon)

    epsilon_policy = (1 - epsilon)*optimal_policy + epsilon*random_policy

    return epsilon_policy
end

function get_T_ss(S, epsilon_policy)

    # get probability of next state given current state given policy... 
    # store in matrix with current state as row, next state as column
    
    NS = length(S)
    N_Objects = length(S[1])

    T_ss = zeros(NS,NS)

    for s_idx = 1:NS
        
        s = S[s_idx]

        legal_actions = findall(s.>0)
        N_legal_actions = length(legal_actions)

        # get probability of next state (row) given legal action (col) 
        prob_S_prime_index_given_action = zeros(NS,N_legal_actions)

        for a_idx = 1:N_legal_actions
            
            A = legal_actions[a_idx]

            possible_S_prime, prob_S_prime = get_possible_s_prime_and_probs(collect(s),A)
            possible_S_prime_idxs = [get_state_index(tuple(this_s...),S) for this_s in possible_S_prime]

            prob_S_prime_index_given_action[possible_S_prime_idxs,a_idx] = prob_S_prime
            
        end

        # marginalize this over probability of actions given policy
        prob_action = epsilon_policy[s_idx,legal_actions]

        prob_S_prime_index = prob_S_prime_index_given_action*prob_action

        # store in T_ss
        T_ss[s_idx,:] = prob_S_prime_index

    end
    
    return T_ss
    
end


##### Simualte the policy...
function simulate_episode(policy, epsilon, NT, S; s = 0)
    
    state_history = zeros(NT, 3)
    action_history = zeros(NT)
    
    NS = length(S)

    if s == 0
        s = sample(S)
    end
    
    N_Objects = length(s)

    for t in 1:NT

        # store the state
        state_history[t,:] .= s

        # select action and transition
        state_idx = get_state_index(tuple(s...), S)

        A = sample(1:N_Objects, ProbabilityWeights(policy[state_idx,:]))

        if rand() < epsilon
            A = sample(findall(s.>0))
        end

        action_history[t] = A

        possible_S_prime, prob_S_prime = get_possible_s_prime_and_probs(collect(s),A)

        s = sample(possible_S_prime, ProbabilityWeights(prob_S_prime))
        
    end
    
    return state_history, action_history
    
end

# get probability remember for N_Timsteps, from start_state_dist
function get_prob_remember_over_time(S, start_state_dist, N_TimeSteps, T_ss)

    S_arr = [collect(s) for s in S]
    pr = [prob_remember(s) for s in S_arr];
    
    N_Objects = length(S[1])

    prob_remember_object = zeros(N_TimeSteps,N_Objects)

    for t = 1:N_TimeSteps
        prob_state_idx =(T_ss^(t-1))'*start_state_dist
        prob_remember_object[t,:] = pr'*prob_state_idx
    end
   
    return prob_remember_object
    
end


## simualte delayed memory test computing prob remember using transitions
function simulate_delayed_memory(N_Quanta, N_Objects, epsilon, N_TimeSteps)
        
    # equal likely object probing
    object_probe_probs = 1/N_Objects*ones(N_Objects)

    # specify reward distr
    exp_num_time_steps = 10
    per_timestep_probe_prob = 1/exp_num_time_steps

    # get all states
    
    print("Generating All States")
    print("\n")
    
    # can this be sped up?
    S = generate_all_states(N_Quanta,N_Objects)
    NS = length(S)
    
    # Computing 
    
    print("Computing Optimal Policy")
    print("\n")
    
    optimal_policy, V = value_iteration(S,object_probe_probs, per_timestep_probe_prob);
    
    random_policy = get_random_quanta_policy(S);
    epsilon_policy = get_epsilon_policy(optimal_policy,random_policy,epsilon);

    # Get state-state transition distribution
    T_ss = get_T_ss(S, epsilon_policy)
    start_state_dist = ones(NS)/NS

    print("Simulating Episode")
    print("\n")
    prob_remember_object = get_prob_remember_over_time(S, start_state_dist, N_TimeSteps, T_ss)
    
    return prob_remember_object
    
end

## simualte delayed memory test computing prob remember using transitions
function simulate_precue(N_Quanta, N_Objects, epsilon, N_TimeSteps; cue_reliability = .8)
        
    object_probe_probs = zeros(N_Objects)
    object_probe_probs[1] = cue_reliability
    object_probe_probs[2:end] .= (1 - cue_reliability)/(N_Objects - 1)
    
    # specify reward distr
    exp_num_time_steps = 10
    per_timestep_probe_prob = 1/exp_num_time_steps

    # get all states
    
    print("Generating All States")
    print("\n")
    
    # can this be sped up?
    S = generate_all_states(N_Quanta,N_Objects)
    NS = length(S)
    
    # Computing 
    
    print("Computing Optimal Policy")
    print("\n")
    
    optimal_policy, V = value_iteration(S,object_probe_probs, per_timestep_probe_prob);
    
    random_policy = get_random_quanta_policy(S);
    epsilon_policy = get_epsilon_policy(optimal_policy,random_policy,epsilon);

    # Get state-state transition distribution
    T_ss = get_T_ss(S, epsilon_policy)
    start_state_dist = ones(NS)/NS

    print("Simulating Episode")
    print("\n")
    prob_remember_object = get_prob_remember_over_time(S, start_state_dist, N_TimeSteps, T_ss)
    
    return prob_remember_object
    
end


# simulate retro-cue paradigm, computing prob remember using transitions
function simulate_retrocue(N_Quanta, N_Objects, epsilon, N_TimeSteps_Pre, N_TimeSteps_Post)
        
    # specify reward distr
    exp_num_time_steps = 10
    per_timestep_probe_prob = 1/exp_num_time_steps

    # get all states
    S = generate_all_states(N_Quanta,N_Objects)
    NS = length(S)

    # change policy to probabilistic
    policy_pre, V = value_iteration(S,1/N_Objects*ones(N_Objects), per_timestep_probe_prob);
    
    obj_probe_probs = zeros(N_Objects)
    obj_probe_probs[1] = .8
    obj_probe_probs[2:end] .= .2/(N_Objects - 1)
    
    policy_post, V = value_iteration(S,obj_probe_probs, per_timestep_probe_prob);
    random_policy = get_random_quanta_policy(S);

    epsilon_policy_pre = get_epsilon_policy(policy_pre,random_policy,epsilon);
    epsilon_policy_post = get_epsilon_policy(policy_post,random_policy,epsilon);

    T_ss_pre = get_T_ss(S, epsilon_policy_pre);
    T_ss_post = get_T_ss(S, epsilon_policy_post);

    # simulate pre-cue part...
    start_state_dist_pre = ones(NS)/NS
    prob_remember_object_pre = get_prob_remember_over_time(S, start_state_dist_pre, N_TimeSteps_Pre, T_ss_pre);

    # here's the post distribution 
    start_state_dist_post = (T_ss_pre^(N_TimeSteps_Pre-1))'*start_state_dist_pre
    prob_remember_object_post = get_prob_remember_over_time(S, start_state_dist_post, N_TimeSteps_Post, T_ss_post);

    prob_remember_object_uncued = [prob_remember_object_pre[:,2]; prob_remember_object_post[:,2]]
    prob_remember_object_cued = [prob_remember_object_pre[:,1]; prob_remember_object_post[:,1]]
    
    return prob_remember_object_cued, prob_remember_object_uncued
    
end
