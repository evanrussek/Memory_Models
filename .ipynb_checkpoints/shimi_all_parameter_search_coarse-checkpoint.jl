using JLD2
include("MDPModelFunctions2.jl")

# array job stuff
is_array_job = true
run_idx = is_array_job ? parse(Int, ENV["SLURM_ARRAY_TASK_ID"]) : 1
on_cluster = true

if on_cluster
    to_save_folder = "/home/erussek/projects/Memory_Models/shimi_all_parameter_search_coarse3"
else
    to_save_folder = "/Users/erussek/Dropbox/Griffiths_Lab_Stuff/code/Memory_Models/shimi_all_parameter_search_coarse3"
end

mkpath(joinpath(to_save_folder,"exp1"))
mkpath(joinpath(to_save_folder,"exp2"))
mkpath(joinpath(to_save_folder,"exp3"))

## Specify parameters for each job...
# 21 epsilon values

# eps_vals = round.(1 .- collect(.01 .* (1.5 .^ (0:2:10))), digits = 3)
eps_vals = collect(.99:-.05:.01)
# quanta values
# q_vals = ceil.(2 .^ collect(1:8))

q_vals = collect(2:5:80)

mem_slopes = [.025, .05, .1, .2, .4]

# re-run with new values so we can see some concavity...
NT_vals = [25, 50, 100, 200, 400, 800] # run w these now... 

job_eps = []
job_q = []
job_nt = []
job_mem_slopes = []

for ms in mem_slopes
    for ep in eps_vals
        for q in q_vals
            for nt in NT_vals

                push!(job_eps, ep)
                push!(job_q, q)
                push!(job_nt, nt)
                push!(job_mem_slopes, ms)

            end
        end
    end
end

n_jobs_total = length(job_nt)

println("N_Jobs_Total: $n_jobs_total")

n_jobs_per_run = 32 # should be about 30 minutes...

n_runs = Int(ceil(n_jobs_total/n_jobs_per_run))
println("N_Runs: $n_runs")

# run_job_idx = 

jobs_per_run = reshape(Vector(1:(n_jobs_per_run*n_runs)), (n_jobs_per_run, n_runs))'

these_jobs = jobs_per_run[run_idx,:]

N_Trials = 1000 # consider making this 1000

for this_job_idx in these_jobs

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


