############################
# LOAD PACKAGES
############################

using JLD2
using DataFrames
using MLUtils
using Flux
using MLUtils
using Statistics
using Random
using ProgressMeter
using Distributions


#### Enter all values at which simulations have been run and saved 

# Epsilon values 
eps_vals = collect(1:-.02:.01) # 17
# eps_vals = eps_vals[eps_vals .> .2]
N_eps = length(eps_vals)

# Quanta values
q_vals = collect(2:4:100) #
N_q = length(q_vals)

# NT vals
NT_vals = [25,50,100,200,400, 800]
N_NT = length(NT_vals)

# Mem Slope Vals
# memory slopes (this won't be looped over)
mem_slopes = [.025, .05, .1, .2, .4, .8, 1.6]
N_ms = length(mem_slopes)

# Task parameters
N_Object_Vals = [2, 3, 4, 5, 6, 8, 10];
N_N_Object_vals = length(N_Object_Vals)

# Other values are time before cue and time after cue
N_Seconds_Pre_Vals = .25 * 2 .^ (0:5)
N_N_Seconds_Pre_Vals = length(N_Seconds_Pre_Vals)

N_Seconds_Post_Vals = .25 * 2 .^ (0:5)
N_N_Seconds_Post_Vals = length(N_Seconds_Post_Vals)

# For trials with no retro-cue - start from .25 seconds and multiply by 1.5 8 times
N_Seconds_NoCue = .25 * 2 .^ (0:6)
N_N_Seconds_NoCue = length(N_Seconds_NoCue)


####### Functions
# Create list of all parameter combinations
function create_parameter_list(eps_vals, q_vals, NT_vals, mem_slopes)
    all_params = []
    
    for epsilon in eps_vals
        for N_Quanta in q_vals
            for NT_per_Second in NT_vals
                for mem_slope in mem_slopes
                    push!(all_params, Dict(
                        "epsilon" => epsilon,
                        "N_Quanta" => Int(N_Quanta),
                        "NT_per_Second" => Int(NT_per_Second),
                        "mem_slope" => mem_slope
                    ))
                end
            end
        end
    end
    
    return all_params
end

function load_results(N_Quanta, epsilon, NT_per_Second; res_part = "delayed_memory")

    # res_part should either be "delayed_memory" or "retrocue"

    res_folder = "/Users/erussek/Dropbox/Griffiths_Lab_Stuff/Data/Memory_Models/manyloadsim_run1"
    
    file_name = "N_Quanta_$(N_Quanta)_epsilon_$(epsilon)_NT_$(NT_per_Second).jld2"

    full_file_path = joinpath(res_folder, res_part, file_name)

    results = load(full_file_path)
    
    return results

end

# function to load delay_prob_correct and retro_cue_prob_correct
function load_prob_correct(N_Quanta, epsilon, NT_per_Second)

    # load delayed memory results
    these_res = load_results(N_Quanta, epsilon, NT_per_Second, res_part = "delayed_memory")

    # N_Obj X N_Sec, N_Memslopes - delay_prob_correct
    delay_prob_correct = these_res["delay_prob_correct"]

    # load retro cue results
    these_res = load_results(N_Quanta, epsilon, NT_per_Second, res_part = "retrocue")

    # N_Obj, N_Sec_Pre, N_Sec_Post, N_Memslopes - retro_cue_prob_correct
    retro_cue_prob_correct = these_res["retro_cue_prob_correct"]

    return delay_prob_correct, retro_cue_prob_correct

end

# Function to load a single data point and turn it into a feature and target vector
function load_data_point(params)
    delay_prob, retro_prob = load_prob_correct(
        params["N_Quanta"],
        params["epsilon"],
        params["NT_per_Second"]
    )

    mem_slopes = [.025, .05, .1, .2, .4, .8, 1.6]
    mem_slope_idx = findfirst(x -> isapprox(x, params["mem_slope"], rtol=1e-10), mem_slopes)
    
    if isnothing(mem_slope_idx)
        error("Invalid mem_slope value: $(params["mem_slope"])")
    end
    
    # Extract and flatten features
    delay_features = vec(delay_prob[:, :, mem_slope_idx])
    retro_features = vec(retro_prob[:, :, :, mem_slope_idx])
    
    # Concatenate features
    features = Float32.(vcat(delay_features, retro_features))
    
    # Target values
    targets = Float32[
        params["N_Quanta"] / 100,
        params["epsilon"],
        params["NT_per_Second"] / 800,
        params["mem_slope"]
    ]
    
    return features, targets
end

function plot_memory(delay_vals, retro_vals, N_Object_Vals, N_Seconds_NoCue, N_Seconds_Pre_Vals, N_Seconds_Post_Vals)
    N_N_Object_vals = length(N_Object_Vals)
    N_N_Seconds_NoCue = length(N_Seconds_NoCue)
    N_N_Seconds_Pre_Vals = length(N_Seconds_Pre_Vals)
    N_N_Seconds_Post_Vals = length(N_Seconds_Post_Vals)

    fig, ax = subplots(1, N_N_Object_vals, figsize=(8, 3), dpi=200, constrained_layout=true, sharex = true, sharey = true)
    for N_Obj_idx = 1:N_N_Object_vals
        ax[N_Obj_idx-1].plot(N_Seconds_NoCue, delay_vals[N_Obj_idx, :], "--o", color = "gray", ms = 3)

        for N_Sec_Pre_idx = 1:length(N_Seconds_Pre_Vals)
            N_Seconds_Pre = N_Seconds_Pre_Vals[N_Sec_Pre_idx]
            ax[N_Obj_idx-1].plot(N_Seconds_Pre .+ N_Seconds_Post_Vals, retro_vals[N_Obj_idx, N_Sec_Pre_idx, :], "--o", ms = 3, color = "gray", label = "$(round(N_Seconds_Pre, digits = 2))")
        end

        ax[N_Obj_idx-1].set_title("N Objects: $(N_Object_Vals[N_Obj_idx])")
        ax[N_Obj_idx-1].set_xlabel("Time (s)")
        ax[N_Obj_idx-1].set_ylabel("Prob. Correct")

    end
end


# build function to turn features back into delay_prob_correct and retro_cue_prob_correct at a single mem_slope_val
function features_to_prob_correct(features, N_Object_Vals, N_Seconds_NoCue, N_Seconds_Pre_Vals, N_Seconds_Post_Vals)
    N_N_Object_vals = length(N_Object_Vals)
    N_N_Seconds_NoCue = length(N_Seconds_NoCue)
    N_N_Seconds_Pre_Vals = length(N_Seconds_Pre_Vals)
    N_N_Seconds_Post_Vals = length(N_Seconds_Post_Vals)

    N_delay_prob_values = N_N_Object_vals * N_N_Seconds_NoCue
    delay_features_vals = features[1:N_delay_prob_values]
    delay_prob_correct = reshape(delay_features_vals, N_N_Object_vals, N_N_Seconds_NoCue)

    N_retro_prob_values = N_N_Object_vals * N_N_Seconds_Pre_Vals * N_N_Seconds_Post_Vals
    retro_features_vals = features[N_delay_prob_values+1:end]
    retro_cue_prob_correct = reshape(retro_features_vals, N_N_Object_vals, N_N_Seconds_Pre_Vals, N_N_Seconds_Post_Vals)

    return delay_prob_correct, retro_cue_prob_correct
end

function collect_all_features_to_matrix(params_list; y_idxs = 1:2) # adjust y_idxs later
    all_features = Vector{Vector{Float32}}()
    all_targets = Vector{Vector{Float32}}()
    
    for params in params_list
        features, targets = load_data_point(params)
        push!(all_features, features)
        push!(all_targets, targets)
    end
    
    # X is n_features X n_samples
    X = reduce(hcat, all_features)

    # Y is n_targets X n_samples
    Y = reduce(hcat, all_targets)
    Y = Y[y_idxs, :];

    return X, Y
end

# Create a data loading function that returns an iterator
function create_data_iterator(X,Y, batch_size)

    # X, Y = collect_all_features_to_matrix(params_list)
    
    # Create DataLoader with matrices
    return DataLoader((X, Y); 
                     batchsize=batch_size, 
                     shuffle=true)
end

function generate_binomial_samples(X_test, N_flips)
    # Get dimensions of X_test
    rows, cols = size(X_test)
    
    # Create output matrix
    samples = zeros(Int, rows, cols)
    
    # Generate binomial samples for each element
    for i in 1:rows
        for j in 1:cols
            p = X_test[i,j]
            samples[i,j] = rand(Binomial(N_flips, p))
        end
    end
    
    return samples ./ N_flips
end


function build_data_matrices(eps_vals, q_vals, NT_vals, mem_slope_vals)
    
        params_list = create_parameter_list(eps_vals, q_vals, NT_vals, mem_slope_vals);

        if length(NT_vals) == 1 && length(mem_slope_vals) == 1
            (X, Y) = collect_all_features_to_matrix(params_list, y_idxs = 1:2); # returns X, Y
        elseif length(mem_slope_vals) == 1
            (X, Y) = collect_all_features_to_matrix(params_list, y_idxs = 1:3); # returns X, Y
        elseif length(NT_vals) == 1
            (X, Y) = collect_all_features_to_matrix(params_list, y_idxs = [1; 2; 4]); # returns X, Y
        else
            (X, Y) = collect_all_features_to_matrix(params_list, y_idxs = 1:4); # returns X, Y
        end
    
        return X, Y
end
    
    
function prep_data_train_test(eps_vals, q_vals, NT_vals, mem_slope_vals; train_prop = 0.7, batch_size = 32)

    X, Y = build_data_matrices(eps_vals, q_vals, NT_vals, mem_slope_vals)

    Xs, Ys = shuffleobs((X, Y))
    (X_train, Y_train), (X_test, Y_test) = splitobs((Xs, Ys); at=train_prop)
    data_loader = create_data_iterator(X_train, Y_train, 32)
    
    return data_loader, X_test, Y_test

end


# train model
function train_model(hidden_size, learning_rate, n_epochs, data_loader, X_test, Y_test)

    loss_fun = Flux.mse;
    
    example_X, example_Y = first(data_loader)

    model = Chain(
        Dense(size(example_X, 1), hidden_size, relu),
        Dense(hidden_size, size(example_Y, 1))
    )

    optim = Flux.setup(Flux.Adam(learning_rate), model) 

    test_losses = []

    @showprogress for epoch in 1:n_epochs
        for (x, y) in data_loader
            loss, grads = Flux.withgradient(model) do m
                # Evaluate model and loss inside gradient context:
                y_hat = m(x)
                loss_fun(y_hat, y)
            end
            Flux.update!(optim, model, grads[1])
            #push!(losses, loss)  # logging, outside gradient context
        end
        test_loss = loss_fun(model(X_test), Y_test)
        push!(test_losses, test_loss)
    end

    return model, test_losses

end




