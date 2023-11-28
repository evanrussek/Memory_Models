using JLD2
include("MDPModelFunctions2.jl")

# array job stuff
is_array_job = false
run_idx = is_array_job ? parse(Int, ENV["SLURM_ARRAY_TASK_ID"]) : 4

on_cluster = false

if on_cluster
    to_save_folder = "/home/erussek/projects/Memory_Models/shimi_1_run1"
else
    to_save_folder = "/Users/erussek/Dropbox/Griffiths_Lab_Stuff/code/Memory_Models/shimi_1_run1"
end

mkpath(to_save_folder)

## Specify parameters for each job...
# 21 epsilon values
eps_vals = 0:.05:1

# quanta values
q_vals = 2:2:40

# 
NT_vals = [25, 50, 100, 200, 400, 800]

job_eps = []
job_q = []
job_nt = []

for ep in eps_vals
    for q in q_vals
        for nt in NT_vals
            push!(job_eps, ep)
            push!(job_q, q)
            push!(job_nt, nt)
        end
    end
end

n_jobs_total = length(job_nt)

println("N_Jobs_Total: $n_jobs_total")

n_jobs_per_run = 360 # should be about 30 minutes...

n_runs = Int(ceil(n_jobs_total/n_jobs_per_run))

# run_job_idx = 

jobs_per_run = reshape(Vector(1:(n_jobs_per_run*n_runs)), (n_jobs_per_run, n_runs))'

these_jobs = jobs_per_run[run_idx,:]

for this_job_idx in these_jobs

    local N_Quanta = job_q[this_job_idx]
    local epsilon = job_eps[this_job_idx]
    local NT_per_Second = job_nt[this_job_idx]

    println("Job: $this_job_idx, N_Quanta: $N_Quanta, epsilon: $epsilon, NT_per_Second: $NT_per_Second")

    local job_res = sim_exp1(epsilon, N_Quanta, NT_per_Second);

    local file_name = "Job_$(this_job_idx)_N_Quanta_$(N_Quanta)_epsilon_$(epsilon)_NT_per_Second_$(NT_per_Second).jld2"
    local full_file_path = joinpath(to_save_folder,file_name)

    jldsave(full_file_path; job_res)
    
end


