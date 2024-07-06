include("MDPModelFunctions.jl")
using Random
using StatsBase

function sample_state(N_Objects, N_Quanta)
    
    # should just pick an assignment for each quanta...
    
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

function simulate_delayed_memory_episode(N_Quanta, N_Objects, epsilon, N_TimeSteps_Pre, N_TimeSteps_Post; cue_reliability = "none")
    
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

function simulate_precue_episode_cowan(N_Quanta, N_Objects_Total, epsilon, N_TimeSteps_Pre, N_TimeSteps_Post; cue_reliability = 1)
    
    
    # N_Objects_Total = 2*N_Objects_Per_Shape
    N_Objects = N_Objects_Total
    
    N_Objects_Per_Shape = Int(N_Objects_Total/2)
    
    object_probe_probs = zeros(N_Objects_Total)
    object_probe_probs[1:N_Objects_Per_Shape] .= cue_reliability/N_Objects_Per_Shape
    
    object_probe_probs[N_Objects_Per_Shape+1:end] .= (1 - cue_reliability)/N_Objects_Per_Shape
    
   # print(object_probe_probs)
   # print(N_B)
    
    # specify reward distr
    exp_num_time_steps = 10
    per_timestep_probe_prob = 1/exp_num_time_steps
    
    state_history, action_history = simulate_episode(N_Quanta, N_Objects, epsilon, N_TimeSteps_Post, object_probe_probs; s=0)
    
    return state_history
    
end


function simulate_task(N_Quanta, N_Objects, epsilon, N_TimeSteps_Pre, N_TimeSteps_Post, N_Trials, sim_episode_fun; mem_slope = .1)
    
    N_TimeSteps = N_TimeSteps_Pre + N_TimeSteps_Post
    prob_remember_all = zeros(N_TimeSteps, N_Objects, N_Trials)
    
    for t in 1:N_Trials
        state_history = sim_episode_fun(N_Quanta, N_Objects, epsilon, N_TimeSteps_Pre, N_TimeSteps_Post)
        prob_remember_all[:, :, t] = prob_remember.(state_history; mem_slope = mem_slope)
        # if you saved this, you could fit the mem_slope param better... 
    end
    
    return dropdims(mean(prob_remember_all, dims=3), dims=3)
    
end

function simulate_task_mult_ms(N_Quanta, N_Objects, epsilon, N_TimeSteps_Pre, N_TimeSteps_Post, N_Trials, sim_episode_fun; mem_slopes = [.05, .1, .15, .2, .25], cue_reliability = 1, baseline_prob = .5)
    
    n_MS = length(mem_slopes)
    
    N_TimeSteps = N_TimeSteps_Pre + N_TimeSteps_Post
    prob_remember_all = zeros(N_TimeSteps, N_Objects, n_MS, N_Trials)
    
    for t in 1:N_Trials
        state_history = sim_episode_fun(N_Quanta, N_Objects, epsilon, N_TimeSteps_Pre, N_TimeSteps_Post; cue_reliability = cue_reliability)
        
        for ms_idx = 1:n_MS
            prob_remember_all[:, :, ms_idx, t] = prob_remember.(state_history; mem_slope = mem_slopes[ms_idx], baseline_prob = baseline_prob)
        end
        # if you saved this, you could fit the mem_slope param better... 
    end

    # take mean over trials
    return dropdims(mean(prob_remember_all, dims=4), dims=4)
    
end



function simulate_task_return_state_hist(N_Quanta, N_Objects, epsilon, N_TimeSteps_Pre, N_TimeSteps_Post, N_Trials, sim_episode_fun; cue_reliability = 1)
    
    N_TimeSteps = N_TimeSteps_Pre + N_TimeSteps_Post
    state_history_all = zeros(N_TimeSteps, N_Objects, N_Trials)
    
    for t in 1:N_Trials
        state_history = sim_episode_fun(N_Quanta, N_Objects, epsilon, N_TimeSteps_Pre, N_TimeSteps_Post; cue_reliability = cue_reliability)
        state_history_all[:, :, t] = state_history;
        # if you saved this, you could fit the mem_slope param easily...  
    end
    
    return state_history_all#dropdims(mean(prob_remember_all, dims=3), dims=3)
    
end

function cowan_k(p_corr, num_obj)
   return num_obj .* (p_corr - (1 .- p_corr))
end



function sim_exp1(epsilon, N_Quanta, NT_per_Second; mem_slope = .1, return_last_only=true, N_Trials = 500)
    
    """
    d_all: is prob correct over time for 3 consitions
    row 1: precue
    row2: neutral
    row3: retrocue
    """
        
    N_Object_Vals = [2,4]

    # these are each 1 second, but question is how many model time-steps occur per real world second...
    N_TimeSteps_Pre = NT_per_Second
    N_TimeSteps_Post = NT_per_Second
    N_TimeSteps = N_TimeSteps_Pre + N_TimeSteps_Post

    d_all = zeros(3, length(N_Object_Vals), N_TimeSteps)

    for (obj_idx, N_Objects) in enumerate(N_Object_Vals)

        # pre-cue all time-steps are post
        d_precue = simulate_task(N_Quanta, N_Objects, epsilon, 0, N_TimeSteps, N_Trials, simulate_precue_episode; mem_slope = mem_slope);
        d_precue = d_precue[:,1]
        d_all[1,obj_idx,:] = d_precue

        # delayed memory - all time-steps are pre
        d_neutral = simulate_task(N_Quanta, N_Objects, epsilon, N_TimeSteps, 0, N_Trials, simulate_delayed_memory_episode; mem_slope = mem_slope);
        d_neutral = d_neutral[:,1]
        d_all[2,obj_idx,:] = d_neutral
        
        # retro-cue
        d_retro = simulate_task(N_Quanta, N_Objects, epsilon, N_TimeSteps_Pre, N_TimeSteps_Post, N_Trials, simulate_retrocue_episode; mem_slope = mem_slope);
        d_retro = d_retro[:,1]
        d_all[3,obj_idx,:] = d_retro

    end
    
    if return_last_only
        return d_all[:,:,end]
    else    
        return d_all
    end
    
end

function sim_exp2(epsilon, N_Quanta, NT_per_Second; mem_slope = .1, return_last_only=true, N_Trials = 500)


    N_Objects = 4

    # do the IM Block

    N_TimeSteps_Pre = Int(round(.2*NT_per_Second))
    N_TimeSteps_Post = Int(round(.5*NT_per_Second))
    N_TimeSteps = N_TimeSteps_Pre + N_TimeSteps_Post

    d_neutral = simulate_task(N_Quanta, N_Objects, epsilon, N_TimeSteps, 0, N_Trials, simulate_delayed_memory_episode; mem_slope = mem_slope);
    p_short_neutral = d_neutral[:,1]

    d_retro = simulate_task(N_Quanta, N_Objects, epsilon, N_TimeSteps_Pre, N_TimeSteps_Post, N_Trials, simulate_retrocue_episode; mem_slope = mem_slope);
    p_short_retro = d_retro[:,1]

    # VSTM
    # Presentation -> 1000ms -> 500 ms

    N_TimeSteps_Pre = Int(round(1*NT_per_Second))
    N_TimeSteps_Post = Int(round(.5*NT_per_Second))
    N_TimeSteps = N_TimeSteps_Pre + N_TimeSteps_Post

    d_neutral = simulate_task(N_Quanta, N_Objects, epsilon, N_TimeSteps, 0, N_Trials, simulate_delayed_memory_episode; mem_slope = mem_slope);
    p_long_neutral = d_neutral[:,1]

    d_retro = simulate_task(N_Quanta, N_Objects, epsilon, N_TimeSteps_Pre, N_TimeSteps_Post, N_Trials, simulate_retrocue_episode; mem_slope = mem_slope);
    p_long_retro = d_retro[:,1];
    
    if return_last_only
        return (p_short_neutral[end], p_short_retro[end], p_long_neutral[end], p_long_retro[end])
    else
        return (p_short_neutral, p_short_retro, p_long_neutral, p_long_retro)
    end
    
end

function sim_exp3(epsilon, N_Quanta, NT_per_Second; mem_slope = .1, return_last_only=true, N_Trials = 500)

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

        d_neutral = simulate_task(N_Quanta, N_Objects, epsilon, N_TimeSteps_IM, 0, N_Trials, simulate_delayed_memory_episode; mem_slope = mem_slope);
        p_IM_neutral[obj_idx,:] = d_neutral[:,1]

        d_retro = simulate_task(N_Quanta, N_Objects, epsilon, N_TimeSteps_Pre_IM, N_TimeSteps_Post_IM, N_Trials, simulate_retrocue_episode; mem_slope = mem_slope);
        p_IM_retro[obj_idx,:] = d_retro[:,1]

        # VSTM
        # Presentation -> 1000ms -> 500 ms

        N_TimeSteps_Pre = Int(round(1*NT_per_Second))
        N_TimeSteps_Post = Int(round(.5*NT_per_Second))
        N_TimeSteps = N_TimeSteps_Pre + N_TimeSteps_Post

        d_neutral = simulate_task(N_Quanta, N_Objects, epsilon, N_TimeSteps_VSTM, 0, N_Trials, simulate_delayed_memory_episode; mem_slope = mem_slope);
        p_VSTM_neutral[obj_idx,:] = d_neutral[:,1]

        d_retro = simulate_task(N_Quanta, N_Objects, epsilon, N_TimeSteps_Pre_VSTM, N_TimeSteps_Post_VSTM, N_Trials, simulate_retrocue_episode; mem_slope = mem_slope);
        p_VSTM_retro[obj_idx,:] = d_retro[:,1];

        # VSTM_long
        # Presentation -> 1800ms -> 500 ms

        d_neutral = simulate_task(N_Quanta, N_Objects, epsilon, N_TimeSteps_Long_VSTM, 0, N_Trials, simulate_delayed_memory_episode; mem_slope = mem_slope);
        p_Long_VSTM_neutral[obj_idx,:] = d_neutral[:,1]

        d_retro = simulate_task(N_Quanta, N_Objects, epsilon, N_TimeSteps_Pre_Long_VSTM, N_TimeSteps_Post_Long_VSTM, N_Trials, simulate_retrocue_episode; mem_slope = mem_slope);
        p_Long_VSTM_retro[obj_idx,:] = d_retro[:,1];

    end
    
    if return_last_only
        return (p_IM_neutral[:,end], p_IM_retro[:,end], p_VSTM_neutral[:,end], p_VSTM_retro[:,end], p_Long_VSTM_neutral[:,end], p_Long_VSTM_retro[:,end])
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

function sim_tanoue_exp1(epsilon, N_Quanta, NT_per_Second; mem_slope = .1, return_last_only=true, N_Trials = 1000)

    N_Objects = 4
    N_TimeSteps_Pre = Int(round(1*NT_per_Second))
    N_TimeSteps_Post_all = [Int(round(x*NT_per_Second)) for x in .1:.1:.7]

    N_TimeSteps_all = N_TimeSteps_Pre .+ N_TimeSteps_Post_all

    p_neutral = simulate_task(N_Quanta, N_Objects, epsilon, N_TimeSteps_all[end], 0, N_Trials, simulate_delayed_memory_episode; mem_slope = mem_slope);
    p_retro = simulate_task(N_Quanta, N_Objects, epsilon, N_TimeSteps_Pre, N_TimeSteps_Post_all[end], N_Trials, simulate_retrocue_episode; mem_slope = mem_slope);
    
    if return_last_only

        p_neutral_res = p_neutral[N_TimeSteps_all,1]
        p_retro_res = p_retro[N_TimeSteps_all,1]
        
    else
        
        p_neutral_res = p_neutral[:,1]
        p_retro_res = p_retro[:,1]
        
    end
    
    return [p_neutral_res; p_retro_res]
    
end

function sim_cowan_1_shape(N_Quanta, epsilon; mem_slopes = [.1], N_Trials = 1000, max_NT_per_sec = 800, return_full_hist = false)

    State_Hist_1_Shape = Dict()

    State_Hist_1_Shape["N_Trials"] = N_Trials
    State_Hist_1_Shape["mem_slopes"] = mem_slopes

    N_Sec = 2 # includes 500 msec encoding and 1500 msec retrieval
    # max_NT_per_sec = 1000

    N_Obj_Conds = [2,3,4,6]

    N_n_obj_conds = length(N_Obj_Conds)

    for obj_cond = 1:N_n_obj_conds

        N_Objects = N_Obj_Conds[obj_cond]

        cond_name = "$(N_Objects)_Objects"
        #print(cond_name)

        sim_episode_fun = simulate_delayed_memory_episode;

        N_TimeSteps_Pre = max_NT_per_sec*N_Sec
        N_TimeSteps_Post = 0

        # returns N_Timepoints X 
        if return_full_hist
            State_Hist_1_Shape[cond_name] = simulate_task_return_state_hist(N_Quanta, N_Objects, epsilon, N_TimeSteps_Pre, N_TimeSteps_Post, N_Trials, sim_episode_fun);
        else
            # this is actually prob remember for the differnet mem_slopes
            
            P_Rem = simulate_task_mult_ms(N_Quanta, N_Objects, epsilon, N_TimeSteps_Pre, N_TimeSteps_Post, N_Trials, sim_episode_fun; mem_slopes = mem_slopes, baseline_prob = 1/12)
            
            # just save the first one since all are the saem
            State_Hist_1_Shape[cond_name] = P_Rem[:,1,:];

        end
        
    end
    
    return State_Hist_1_Shape
    
end

function sim_cowan_att(N_Quanta, epsilon; N_Trials = 1000, max_NT_per_sec = 800, return_full_hist = false, mem_slopes = [.1])

    # These are all post cue - though sometimes the cue is uninformative... (then this should basically reduce to the one-shape case)
    
    N_Sec = 2 # includes 500 msec encoding and 1500 msec retrieval
    N_TimeSteps_Pre = 0
    N_TimeSteps_Post = max_NT_per_sec*N_Sec
    
    CueR_Conds = [1., .8, .5]
    N_CueR_Conds = length(CueR_Conds)

    N_Obj_Per_Shape_Conds = [2, 3]
    N_N_Obj_Per_Shape_Conds = length(N_Obj_Per_Shape_Conds)

    State_Hist_Att = Dict()

    for Cuer_Cond_Idx = 1:N_CueR_Conds
        for N_Obj_Per_Shape_Cond_Idx = 1:N_N_Obj_Per_Shape_Conds

    # Cuer_Cond_Idx=1
    # N_Obj_Per_Shape_Cond_Idx=2
            CueR = CueR_Conds[Cuer_Cond_Idx]
            N_Obj_Per_Shape = N_Obj_Per_Shape_Conds[N_Obj_Per_Shape_Cond_Idx]

            N_Objects_Total = 2*N_Obj_Per_Shape;

            cond_name = "CueR_$(CueR)_N_Obj_Per_Shape_$(N_Obj_Per_Shape)"
            #print(string(cond_name, " "))

            sim_episode_fun = simulate_precue_episode_cowan;


            if return_full_hist 
                State_Hist_Att[cond_name] = simulate_task_return_state_hist(N_Quanta, N_Objects_Total, epsilon, N_TimeSteps_Pre, N_TimeSteps_Post, N_Trials, sim_episode_fun; cue_reliability = CueR);
            else
                P_Rem = simulate_task_mult_ms(N_Quanta, N_Objects_Total, epsilon, N_TimeSteps_Pre, N_TimeSteps_Post, N_Trials, sim_episode_fun; cue_reliability = CueR, mem_slopes = mem_slopes, baseline_prob = 1/12);
                
                which_obj_idxs = [1, N_Obj_Per_Shape+1]
                State_Hist_Att[cond_name] = P_Rem[:, which_obj_idxs, :];

            end
        end
    end
    
    return State_Hist_Att
    
end



