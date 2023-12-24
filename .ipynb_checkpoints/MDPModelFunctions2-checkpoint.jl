include("MDPModelFunctions.jl")
using Random
using StatsBase

function sample_state(N_Objects, N_Quanta)
    
    # Step 1: Generate N-1 random integers
    random_integers = rand(1:N_Quanta, N_Objects-1)

    # Step 2: Sort the generated integers
    sort!(random_integers)
    
    # Step 3: Calculate the differences
    differences = [random_integers[1]; diff(random_integers); N_Quanta - random_integers[end]]

    return shuffle(differences)
    
end

function get_Pol_onestep_R(s, object_probe_probs, per_timestep_probe_prob, epsilon) # also get the policy?
    
    # could this be biased?
    
    N_Objects = length(s)

    
    legal_actions = findall(s.>0)
    N_legal_actions = length(legal_actions)
    Q = zeros(N_legal_actions)

    for a_idx in 1:N_legal_actions
    # a_idx = 1

        A = legal_actions[a_idx]

        possible_S_prime, prob_S_prime = get_possible_s_prime_and_probs(s,A)
        possible_R_prime = [get_state_reward(this_s, object_probe_probs, per_timestep_probe_prob) for this_s in possible_S_prime]

        Q[a_idx] = prob_S_prime'*possible_R_prime # just look to the next state
    end
    
    Q = round.(Q,digits=6)

    max_val = maximum(Q)

    max_Q_idxs = findall(Q .== max_val)
    max_actions = legal_actions[max_Q_idxs]

    prob_actions_Q = zeros(N_Objects)
    prob_actions_Q[max_actions] .= 1 ./ length(max_actions)     
    
    prob_action_random = s / sum(s) # quanta chosen at random -- could also have states chosen at random
    
    prob_actions = (1 - epsilon)*prob_actions_Q + epsilon*prob_action_random
    
    return prob_actions
end

function simulate_episode(N_Quanta, N_Objects, epsilon, N_TimeSteps, object_probe_probs; s=0)
    
    exp_num_time_steps = 10
    per_timestep_probe_prob = 1/exp_num_time_steps

    state_history = zeros(N_TimeSteps, N_Objects)
    action_history = zeros(N_TimeSteps)

    if s == 0
        s = sample_state(N_Objects, N_Quanta)
    end

    for t in 1:N_TimeSteps

        # store the state
        state_history[t,:] .= s

        prob_actions = get_Pol_onestep_R(s, object_probe_probs, per_timestep_probe_prob, epsilon)

        A = sample(1:N_Objects, ProbabilityWeights(prob_actions))

        action_history[t] = A

        possible_S_prime, prob_S_prime = get_possible_s_prime_and_probs(s,A)

        s = sample(possible_S_prime, ProbabilityWeights(prob_S_prime))

    end

    return state_history, action_history
    
end

function simulate_delayed_memory_episode(N_Quanta, N_Objects, epsilon, N_TimeSteps_Pre, N_TimeSteps_Post)
    
    # equal likely object probing
    object_probe_probs = 1/N_Objects*ones(N_Objects)

    # specify reward distr
    exp_num_time_steps = 10
    per_timestep_probe_prob = 1/exp_num_time_steps
    
    
    state_history, action_history = simulate_episode(N_Quanta, N_Objects, epsilon, N_TimeSteps_Pre, object_probe_probs; s=0)
    
    return state_history
    
end

function simulate_precue_episode(N_Quanta, N_Objects, epsilon, N_TimeSteps_Pre, N_TimeSteps_Post; cue_reliability = 1)
    
    object_probe_probs = zeros(N_Objects)
    object_probe_probs[1] = cue_reliability
    object_probe_probs[2:end] .= (1 - cue_reliability)/(N_Objects - 1)
    
    # specify reward distr
    exp_num_time_steps = 10
    per_timestep_probe_prob = 1/exp_num_time_steps
    
    state_history, action_history = simulate_episode(N_Quanta, N_Objects, epsilon, N_TimeSteps_Post, object_probe_probs; s=0)
    
    return state_history
    
end

function simulate_retrocue_episode(N_Quanta, N_Objects, epsilon, N_TimeSteps_Pre, N_TimeSteps_Post; cue_reliability = 1)
    
    # specify reward distr
    exp_num_time_steps = 10
    per_timestep_probe_prob = 1/exp_num_time_steps
    
    # equal likely object probing
    object_probe_probs = 1/N_Objects*ones(N_Objects)
    state_history_pre, _ = simulate_episode(N_Quanta, N_Objects, epsilon, N_TimeSteps_Pre, object_probe_probs; s=0)
    
        
    object_probe_probs = zeros(N_Objects)
    object_probe_probs[1] = cue_reliability
    object_probe_probs[2:end] .= (1 - cue_reliability)/(N_Objects - 1)
    
    state_history_post, _ = simulate_episode(N_Quanta, N_Objects, epsilon, N_TimeSteps_Post, object_probe_probs; s=state_history_pre[end,:])

    
    return [state_history_pre; state_history_post]
    
end

function simulate_task(N_Quanta, N_Objects, epsilon, N_TimeSteps_Pre, N_TimeSteps_Post, N_Trials, sim_episode_fun)
    
    N_TimeSteps = N_TimeSteps_Pre + N_TimeSteps_Post
    prob_remember_all = zeros(N_TimeSteps, N_Objects, N_Trials)
    
    for t in 1:N_Trials
        state_history = sim_episode_fun(N_Quanta, N_Objects, epsilon, N_TimeSteps_Pre, N_TimeSteps_Post)
        prob_remember_all[:, :, t] = prob_remember.(state_history)
    end
    
    return dropdims(mean(prob_remember_all, dims=3), dims=3)
    
end

function cowan_k(p_corr, num_obj)
   return num_obj .* (p_corr - (1 .- p_corr))
end



function sim_exp1(epsilon, N_Quanta, NT_per_Second; return_last_only=true)
    
    """
    d_all: is prob correct over time for 3 consitions
    row 1: precue
    row2: neutral
    row3: retrocue
    """
    
    N_Trials = 1000;
    
    N_Object_Vals = [2,4]

    # these are each 1 second, but question is how many model time-steps occur per real world second...
    N_TimeSteps_Pre = NT_per_Second
    N_TimeSteps_Post = NT_per_Second
    N_TimeSteps = N_TimeSteps_Pre + N_TimeSteps_Post

    d_all = zeros(3, length(N_Object_Vals), N_TimeSteps)

    for (obj_idx, N_Objects) in enumerate(N_Object_Vals)

        # pre-cue all time-steps are post
        d_precue = simulate_task(N_Quanta, N_Objects, epsilon, 0, N_TimeSteps, N_Trials, simulate_precue_episode);
        d_precue = d_precue[:,1]
        d_all[1,obj_idx,:] = d_precue

        # delayed memory - all time-steps are pre
        d_neutral = simulate_task(N_Quanta, N_Objects, epsilon, N_TimeSteps, 0, N_Trials, simulate_delayed_memory_episode);
        d_neutral = d_neutral[:,1]
        d_all[2,obj_idx,:] = d_neutral
        
        # retro-cue
        d_retro = simulate_task(N_Quanta, N_Objects, epsilon, N_TimeSteps_Pre, N_TimeSteps_Post, N_Trials, simulate_retrocue_episode);
        d_retro = d_retro[:,1]
        d_all[3,obj_idx,:] = d_retro

    end
    
    if return_last_only
        return d_all[:,:,end]
    else    
        return d_all
    end
    
end

function sim_exp2(epsilon, N_Quanta, NT_per_Second; return_last_only=true)

    N_Trials = 1000;

    N_Objects = 4

    # do the IM Block

    N_TimeSteps_Pre = Int(round(.2*NT_per_Second))
    N_TimeSteps_Post = Int(round(.5*NT_per_Second))
    N_TimeSteps = N_TimeSteps_Pre + N_TimeSteps_Post

    d_neutral = simulate_task(N_Quanta, N_Objects, epsilon, N_TimeSteps, 0, N_Trials, simulate_delayed_memory_episode);
    p_short_neutral = d_neutral[:,1]

    d_retro = simulate_task(N_Quanta, N_Objects, epsilon, N_TimeSteps_Pre, N_TimeSteps_Post, N_Trials, simulate_retrocue_episode);
    p_short_retro = d_retro[:,1]

    # VSTM
    # Presentation -> 1000ms -> 500 ms

    N_TimeSteps_Pre = Int(round(1*NT_per_Second))
    N_TimeSteps_Post = Int(round(.5*NT_per_Second))
    N_TimeSteps = N_TimeSteps_Pre + N_TimeSteps_Post

    d_neutral = simulate_task(N_Quanta, N_Objects, epsilon, N_TimeSteps, 0, N_Trials, simulate_delayed_memory_episode);
    p_long_neutral = d_neutral[:,1]

    d_retro = simulate_task(N_Quanta, N_Objects, epsilon, N_TimeSteps_Pre, N_TimeSteps_Post, N_Trials, simulate_retrocue_episode);
    p_long_retro = d_retro[:,1];
    
    if return_last_only
        return (p_short_neutral[end], p_short_retro[end], p_long_neutral[end], p_long_retro[end])
    else
        return (p_short_neutral, p_short_retro, p_long_neutral, p_long_retro)
    end
    
end

function sim_exp3(epsilon, N_Quanta, NT_per_Second; return_last_only=true)

    N_Trials = 1000;

    N_Object_Vals = [3,6]

    N_nobj = length(N_Object_Vals);

    # IM time scales...
    N_TimeSteps_Pre_IM = Int(round(.2*NT_per_Second))
    N_TimeSteps_Post_IM = Int(round(.5*NT_per_Second))
    N_TimeSteps_IM = N_TimeSteps_Pre_IM + N_TimeSteps_Post_IM

    # SHORT Time Scales...
    N_TimeSteps_Pre_VSTM = Int(round(1*NT_per_Second))
    N_TimeSteps_Post_VSTM = Int(round(.5*NT_per_Second))
    N_TimeSteps_VSTM = N_TimeSteps_Pre_VSTM + N_TimeSteps_Post_VSTM

    # long time-scales...
    N_TimeSteps_Pre_Long_VSTM = Int(round(1.8*NT_per_Second))
    N_TimeSteps_Post_Long_VSTM = Int(round(.5*NT_per_Second))
    N_TimeSteps_Long_VSTM = N_TimeSteps_Pre_Long_VSTM + N_TimeSteps_Post_Long_VSTM

    p_IM_neutral = zeros(2, N_TimeSteps_IM)
    p_IM_retro = zeros(2, N_TimeSteps_IM)

    p_VSTM_neutral = zeros(2, N_TimeSteps_VSTM)
    p_VSTM_retro = zeros(2, N_TimeSteps_VSTM)

    p_Long_VSTM_neutral = zeros(2, N_TimeSteps_Long_VSTM)
    p_Long_VSTM_retro = zeros(2, N_TimeSteps_Long_VSTM)

    for (obj_idx, N_Objects) in enumerate(N_Object_Vals)

        # do the IM Block

        d_neutral = simulate_task(N_Quanta, N_Objects, epsilon, N_TimeSteps_IM, 0, N_Trials, simulate_delayed_memory_episode);
        p_IM_neutral[obj_idx,:] = d_neutral[:,1]

        d_retro = simulate_task(N_Quanta, N_Objects, epsilon, N_TimeSteps_Pre_IM, N_TimeSteps_Post_IM, N_Trials, simulate_retrocue_episode);
        p_IM_retro[obj_idx,:] = d_retro[:,1]

        # VSTM
        # Presentation -> 1000ms -> 500 ms

        N_TimeSteps_Pre = Int(round(1*NT_per_Second))
        N_TimeSteps_Post = Int(round(.5*NT_per_Second))
        N_TimeSteps = N_TimeSteps_Pre + N_TimeSteps_Post

        d_neutral = simulate_task(N_Quanta, N_Objects, epsilon, N_TimeSteps_VSTM, 0, N_Trials, simulate_delayed_memory_episode);
        p_VSTM_neutral[obj_idx,:] = d_neutral[:,1]

        d_retro = simulate_task(N_Quanta, N_Objects, epsilon, N_TimeSteps_Pre_VSTM, N_TimeSteps_Post_VSTM, N_Trials, simulate_retrocue_episode);
        p_VSTM_retro[obj_idx,:] = d_retro[:,1];

        # VSTM_long
        # Presentation -> 1800ms -> 500 ms

        d_neutral = simulate_task(N_Quanta, N_Objects, epsilon, N_TimeSteps_Long_VSTM, 0, N_Trials, simulate_delayed_memory_episode);
        p_Long_VSTM_neutral[obj_idx,:] = d_neutral[:,1]

        d_retro = simulate_task(N_Quanta, N_Objects, epsilon, N_TimeSteps_Pre_Long_VSTM, N_TimeSteps_Post_Long_VSTM, N_Trials, simulate_retrocue_episode);
        p_Long_VSTM_retro[obj_idx,:] = d_retro[:,1];

    end
    
    if return_last_only
        return (p_IM_neutral[end], p_IM_retro[end], p_VSTM_neutral[end], p_VSTM_retro[end], p_Long_VSTM_neutral[end], p_Long_VSTM_retro[end])
    else
        return (p_IM_neutral, p_IM_retro, p_VSTM_neutral, p_VSTM_retro, p_Long_VSTM_neutral, p_Long_VSTM_retro)
    end

end

function plot_over_time_exp2(p_short_neutral, p_short_retro, p_long_neutral, p_long_retro, NT_per_Second; title = "")

    fig,ax = subplots(1, 2, figsize = (4,2.5), dpi=200,constrained_layout=true, sharey=true)

    N_TimeSteps_Pre = Int(round(.2*NT_per_Second))
    N_TimeSteps_Post = Int(round(.5*NT_per_Second))
    N_TimeSteps = N_TimeSteps_Pre + N_TimeSteps_Post
    N_TimeSteps_IM = N_TimeSteps

    ax[0].plot(1:N_TimeSteps_IM, cowan_k(p_short_neutral,4))
    ax[0].plot(1:N_TimeSteps_IM, cowan_k(p_short_retro,4))
    ax[0].set_title("IM")

    N_TimeSteps_Pre = Int(round(1*NT_per_Second))
    N_TimeSteps_Post = Int(round(.5*NT_per_Second))
    N_TimeSteps = N_TimeSteps_Pre + N_TimeSteps_Post
    N_TimeSteps_IM = N_TimeSteps

    ax[1].plot(1:N_TimeSteps_IM, cowan_k(p_long_neutral,4))
    ax[1].plot(1:N_TimeSteps_IM, cowan_k(p_long_retro,4))

    ax[0].set_ylabel("Cowan's K")
    ax[0].set_xlabel("Time-Steps")
    ax[1].set_xlabel("Time-Steps")
    ax[1].set_title("VSTM")
    
    fig.suptitle(title)

end

