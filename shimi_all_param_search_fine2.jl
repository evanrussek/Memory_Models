using JLD2
include("MDPModelFunctions2.jl")

# array job stuff
is_array_job = true
run_idx = is_array_job ? parse(Int, ENV["SLURM_ARRAY_TASK_ID"]) : 1
on_cluster = true

if on_cluster
    to_save_folder = "/scratch/gpfs/erussek/Memory_Models/shimi_all_parameter_search_more_vals"
else
    to_save_folder = "/Users/erussek/Dropbox/Griffiths_Lab_Stuff/Data/Memory_Models/shimi_all_parameter_search_fine_nt_25_50"
end

mkpath(joinpath(to_save_folder,"exp1"))
mkpath(joinpath(to_save_folder,"exp2"))
mkpath(joinpath(to_save_folder,"exp3"))

## Specify parameters for each job...
# 21 epsilon values

eps_vals = collect(1:-.01:.01) # 17

# quanta values
q_vals = collect(2:2:80) # could go to one... 

mem_slopes = [.1, .1, .2]

# re-run with new values so we can see some concavity...
NT_vals = [100, 200, 100] # run w these now... 

job_eps = []
job_q = []
job_nt = []
job_mem_slopes = []


for ep in eps_vals
    for q in q_vals
        for hyp_idx in 1:length(mem_slopes)
            
            nt = NT_vals[hyp_idx]
            ms = mem_slopes[hyp_idx]

            push!(job_eps, ep)
            push!(job_q, q)
            push!(job_nt, nt)
            push!(job_mem_slopes, ms)

        end
    end
end


n_jobs_total = length(job_nt)

println("N_Jobs_Total: $n_jobs_total")

n_jobs_per_run = 200 # 
n_runs = Int(ceil(n_jobs_total/n_jobs_per_run))
println("N_Runs: $n_runs")

# run_job_idx = 

jobs_per_run = reshape(Vector(1:(n_jobs_per_run*n_runs)), (n_jobs_per_run, n_runs))'

these_jobs = jobs_per_run[run_idx,:]

N_Trials = 1000 # consider making this 1000

for this_job_idx in these_jobs
    
    if (this_job_idx > n_jobs_total)
        break
    end

    local N_Quanta = job_q[this_job_idx]
    local epsilon = job_eps[this_job_idx]
    local NT_per_Second = job_nt[this_job_idx]
    local mem_slope = job_mem_slopes[this_job_idx]

    println("Job: $this_job_idx, N_Quanta: $N_Quanta, epsilon: $epsilon, NT_per_Second: $NT_per_Second, mem_slope: $mem_slope")
    local file_name = "N_Quanta_$(N_Quanta)_epsilon_$(epsilon)_NT_per_Second_$(NT_per_Second)_memslope_$(mem_slope).jld2"
    
    local job_res_1 = sim_exp1(epsilon, N_Quanta, NT_per_Second; mem_slope = mem_slope, return_last_only=true, N_Trials = N_Trials);
    local full_file_path = joinpath(to_save_folder,"exp1",file_name)
    jldsave(full_file_path; job_res_1)

    local job_res_2 = sim_exp2(epsilon, N_Quanta, NT_per_Second; mem_slope = mem_slope,return_last_only=true, N_Trials = N_Trials);
    local full_file_path = joinpath(to_save_folder,"exp2",file_name)
    jldsave(full_file_path; job_res_2)
    
    local job_res_3 = sim_exp3(epsilon, N_Quanta, NT_per_Second; mem_slope = mem_slope, return_last_only=true, N_Trials = N_Trials);
    local full_file_path = joinpath(to_save_folder,"exp3",file_name)
    jldsave(full_file_path; job_res_3)
    
    # testing this out -- garbage collection
    GC.gc(true)
    
end


