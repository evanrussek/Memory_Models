## This will return full state-histories 
# (so, this is super high memory, but you don't)
# have to loop through hyper-params (can sort out after)

# packages to call
using JLD2
include("MDPModelFunctions2.jl")

# array job stuff
is_array_job = true
run_idx = is_array_job ? parse(Int, ENV["SLURM_ARRAY_TASK_ID"]) : 1
on_cluster = true

if on_cluster
    to_save_folder = "/scratch/gpfs/erussek/Memory_Models/Cowan_Param_Sweep_1"
else
    to_save_folder = "/Users/erussek/Dropbox/Griffiths_Lab_Stuff/Data/Memory_Models/Cowan_Param_Sweep_1"
end

mkpath(joinpath(to_save_folder,"one_shape"))
mkpath(joinpath(to_save_folder,"two_shape"))

# epsilon vals
eps_vals = collect(1:-.02:.01) # 

# vary the mem_slopes
mem_slopes = [.025, .05, .1, .15, .2, .25]

# quanta values
q_vals = collect(2:2:100) # could go to one... 

# build all job parameters
job_eps = []
job_q = []

for ep in eps_vals
    for q in q_vals
        
        push!(job_eps, ep)
        push!(job_q, q)
        
    end
end


n_jobs_total = length(job_eps)

println("N_Jobs_Total: $n_jobs_total")

n_jobs_per_run = 50 # 
n_runs = Int(ceil(n_jobs_total/n_jobs_per_run))
println("N_Runs: $n_runs")

jobs_per_run = reshape(Vector(1:(n_jobs_per_run*n_runs)), (n_jobs_per_run, n_runs))'

these_jobs = jobs_per_run[run_idx,:]

N_Trials = 1000 

for this_job_idx in these_jobs
    
    if (this_job_idx > n_jobs_total)
        break
    end

    local N_Quanta = job_q[this_job_idx]
    local epsilon = job_eps[this_job_idx]

    println("Job: $this_job_idx, N_Quanta: $N_Quanta, epsilon: $epsilon")
    
    local file_name = "N_Quanta_$(N_Quanta)_epsilon_$(epsilon).jld2"
    
    # run the one shape condition and save
    local state_hist_one_shape = sim_cowan_1_shape(N_Quanta, epsilon, mem_slopes = mem_slopes)
    local full_file_path = joinpath(to_save_folder,"one_shape",file_name)    
    jldsave(full_file_path; state_hist_one_shape)
    
    # run the two shape condition
    local state_hist_two_shape = sim_cowan_att(N_Quanta, epsilon, mem_slopes = mem_slopes)
    local full_file_path = joinpath(to_save_folder,"two_shape",file_name)    
    jldsave(full_file_path; state_hist_two_shape)
    
    # testing this out -- garbage collection
    GC.gc(true)
    
end

