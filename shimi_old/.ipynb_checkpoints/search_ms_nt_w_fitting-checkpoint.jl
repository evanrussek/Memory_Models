## started this...

include("MDPModelFunctions2.jl")
using BlackBoxOptim
using JLD2

#Random.seed!(1234);


# array job stuff
is_array_job = true
job_idx = is_array_job ? parse(Int, ENV["SLURM_ARRAY_TASK_ID"]) : 1
on_cluster = true

# here's what we'll grid search over

# re-run with new values so we can see some concavity...
NT_vals = collect(100:250:2200)
mem_slopes = round.([.01*(1.5^x) for x = 2:9], digits = 3)

job_nt = []
job_mem_slopes = []
job_groups = []

for which_group in ["child", "adult"]
    for ms in mem_slopes
        for nt in NT_vals

            push!(job_mem_slopes, ms)
            push!(job_nt, nt)
            push!(job_groups, which_group)

        end
    end
end

n_jobs_total = length(job_nt)

println("N_Jobs_Total: $n_jobs_total")

if on_cluster
    to_save_folder = "/home/erussek/projects/Memory_Models/shimi_bbopts"
else
    to_save_folder = "/Users/erussek/Dropbox/Griffiths_Lab_Stuff/code/Memory_Models/shimi_bbopts"
end

mkpath(to_save_folder)

function format_res_cowan_k(res; exp_num = 1)
    
    if exp_num == 1
    
        # row is 2 obj vs 4 obj
        # col is pre-cue, neutral, retro-cue, neutral
        
        part_pcorr = res;
        model_cowan_p_corr = [part_pcorr; part_pcorr[2,:]']'
        model_cowan_k_mtx = cowan_k(model_cowan_p_corr, 2)
        model_cowan_k_mtx[2,:] = cowan_k(model_cowan_p_corr[2,:], 4)
        
        model_cowan_k = [model_cowan_k_mtx[1,:]' model_cowan_k_mtx[2,:]']';#[model_cowan_k_mtx[1,:] model_cowan_k_mtx[2,:]]
    elseif exp_num == 2
        
        # IM cued, IM neutral, VSTM cued, VSTM neutral - need to adjust from what it was saved as
        
        (p_short_neutral, p_short_retro, p_long_neutral, p_long_retro) = res
        model_cowan_k = cowan_k([p_short_retro p_short_neutral p_long_retro p_long_neutral],4)'
        
    else
        
        (p_IM_neutral, p_IM_retro, p_VSTM_neutral, p_VSTM_retro, p_Long_VSTM_neutral, p_Long_VSTM_retro) = res

        # Load 3 cued, Load 3 Neutral, Load 6 cued, Load 6 neutral
        IM_model = [cowan_k(p_IM_retro[1],3) cowan_k(p_IM_neutral[1],3) cowan_k(p_IM_retro[2],6) cowan_k(p_IM_neutral[2], 6)]
        Short_VSTM_model = [cowan_k(p_VSTM_retro[1],3) cowan_k(p_VSTM_neutral[1],3) cowan_k(p_VSTM_retro[2],6) cowan_k(p_VSTM_neutral[2], 6)]
        Long_VSTM_model = [cowan_k(p_Long_VSTM_retro[1],3) cowan_k(p_Long_VSTM_neutral[1],3) cowan_k(p_Long_VSTM_retro[2],6) cowan_k(p_Long_VSTM_neutral[2], 6)]

        # 3 x 4
        model_cowan_k = [IM_model; Short_VSTM_model; Long_VSTM_model]  
        
    end
        
    return model_cowan_k
    
end    


# 6 parameteres for 44 datapoints 

exp1_cowan_k_true_7_yr = [1.16 1.14 1.13 1.02 2.80 .91 1.30 .96]' # 8
exp1_cowan_k_true_adult = [1.98 1.94 1.97 1.95 3.87 2.66 3.64 2.87]'

# experiment 2 - 4 objects
# col is IM cued, IM neutral, VSTM cued, VSTM neutral

exp2_cowan_k_true_7_yr = [1.55 1.16 1.00 0.69]' # 4
exp2_cowan_k_true_adult = [3.54 2.58 3.58 2.65]'

# experiment 3

# Load 3 cued, Load 3 Neutral, Load 6 cued, Load 6 neutral
IM_true_7_yr = [1.44 1.27 1.82 0.88]
Short_VSTM_true_7_yr = [1.14 0.77 NaN NaN]
Long_VSTM_true_7_yr = [1.19 0.90 NaN NaN]

# 8

exp3_cowan_k_true_7_yr = [IM_true_7_yr; Short_VSTM_true_7_yr; Short_VSTM_true_7_yr]


IM_true_adult = [2.63 2.41 4.80 2.46]
Short_VSTM_true_adult = [2.75 2.38 3.97 2.15]
Long_VSTM_true_adult = [2.76 2.30 3.65 1.75]
# 12

exp3_cowan_k_true_adult = [IM_true_adult; Short_VSTM_true_adult; Short_VSTM_true_adult]

# function to compute mean squared error...

# takes in Adult_Quanta (int), Adult_Epsilon (0-1 bounded), Child_Quanta (int), Child_Epsilon (0-1 bounded), Mem_slope (less than 1), NT_val # 6 params...

# params = [20, .8, 5, .4, .1, 100]

function compute_mse(params, exp1_cowan_k_true_7_yr, exp1_cowan_k_true_adult, exp2_cowan_k_true_7_yr, exp2_cowan_k_true_adult, exp3_cowan_k_true_7_yr, exp3_cowan_k_true_adult; which_group = "adult", mem_slope = .01, nt_val = 100, N_Trials = 500)
    
    
    # garbage collect...
    GC.gc(true)

    if which_group == "adult"
        
        # min number of quanta is 2...
        adult_quanta = 2 + round(100*params[1])
        adult_epsilon = params[2]
        
        adult_res_1 = sim_exp1(adult_epsilon, adult_quanta, nt_val; mem_slope = mem_slope, N_Trials = N_Trials);
        adult_cowan_k1 = format_res_cowan_k(adult_res_1; exp_num = 1)
        adult_res_2 = sim_exp2(adult_epsilon, adult_quanta, nt_val; mem_slope = mem_slope, N_Trials = N_Trials);
        adult_cowan_k2 = format_res_cowan_k(adult_res_2; exp_num = 2)
        adult_res_3 = sim_exp3(adult_epsilon, adult_quanta, nt_val; mem_slope = mem_slope, N_Trials = N_Trials);        
        adult_cowan_k3 = format_res_cowan_k(adult_res_3; exp_num = 3)

        adult_mse1 = sum((adult_cowan_k1 .- exp1_cowan_k_true_adult).^2)

        adult_mse2 = sum((adult_cowan_k2 .- exp2_cowan_k_true_adult).^2)

        adult_mse3 = sum((adult_cowan_k3 .- exp3_cowan_k_true_adult).^2)

        adult_mse = adult_mse1 + adult_mse2 + adult_mse3
        
        this_mse = adult_mse
        
    else
        
        # min is 2
        child_quanta = 2 + round(100*params[1])
        child_epsilon = params[2]
        
        children_res_1 = sim_exp1(child_epsilon, child_quanta, nt_val; mem_slope = mem_slope, N_Trials = N_Trials);
        children_cowan_k1 = format_res_cowan_k(children_res_1; exp_num = 1)
        children_res_2 = sim_exp2(child_epsilon, child_quanta, nt_val; mem_slope = mem_slope, N_Trials = N_Trials);
        children_cowan_k2 = format_res_cowan_k(children_res_2; exp_num = 2)
        children_res_3 = sim_exp3(child_epsilon, child_quanta, nt_val; mem_slope = mem_slope, N_Trials = N_Trials);
        children_cowan_k3 = format_res_cowan_k(children_res_3; exp_num = 3)
        
        child_mse1 = sum((children_cowan_k1 .- exp1_cowan_k_true_7_yr).^2)

        child_mse2 = sum((children_cowan_k2 .- exp2_cowan_k_true_7_yr).^2)

        child_sq_err = (children_cowan_k3[:] .- exp3_cowan_k_true_7_yr[:]).^2
        child_mse3 = sum(child_sq_err[.!isnan.(child_sq_err)])

        child_mse = child_mse1 + child_mse2 + child_mse3
        
        this_mse = child_mse
    end


    #both_mse = child_mse + adult_mse
    
    return this_mse
    
end


ms = job_mem_slopes[job_idx]
nt = job_nt[job_idx]
which_group = job_groups[job_idx]


target(params) = compute_mse(params, exp1_cowan_k_true_7_yr, exp1_cowan_k_true_adult, exp2_cowan_k_true_7_yr, exp2_cowan_k_true_adult, exp3_cowan_k_true_7_yr, exp3_cowan_k_true_adult, which_group = which_group, mem_slope = ms, nt_val = nt, N_Trials = 500)
#res = bboptimize(target; NumDimensions = 2, SearchRange = (0.0,1.0),NThreads=Threads.nthreads()-1, MaxTime=14400.0) # run for 4 hours...

res = bboptimize(target; NumDimensions = 2, SearchRange = (0.0,1.0),MaxTime=10800.0) # run for 3 hours...


best_x = best_candidate(res)
best_val = best_fitness(res)

res_dict = Dict()
res_dict["ms"] = ms
res_dict["nt"] = nt
res_dict["which_group"] = which_group
res_dict["x"] = best_x
res_dict["val"] = best_val

file_name = "Group_$(which_group)_NT_$(nt)_memslope_$(ms).jld2"
full_file_path = joinpath(to_save_folder,file_name)
jldsave(full_file_path; res_dict)



# save the results...