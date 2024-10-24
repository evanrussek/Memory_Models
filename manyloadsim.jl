using JLD2

include("MDPModelFunctions2.jl")

# array job stuff
is_array_job = true
run_idx = is_array_job ? parse(Int, ENV["SLURM_ARRAY_TASK_ID"]) : 1
on_cluster = true

if on_cluster
    to_save_folder = "/scratch/gpfs/erussek/Memory_Models/manyloadsim_run1"
else
    to_save_folder = "/Users/erussek/Dropbox/Griffiths_Lab_Stuff/Data/Memory_Models/manyloadsim_run1"
end

mkpath(joinpath(to_save_folder,"delayed_memory"))
mkpath(joinpath(to_save_folder,"retrocue"))
mkpath(joinpath(to_save_folder,"condition_vals"))


# create  function to simulate delayed forgetting conditions
function simulate_delayed_memory(N_Object_Vals, N_Seconds_NoCue, mem_slopes, N_Quanta, epsilon, NT_per_Sec)
    
    N_N_Object_vals = length(N_Object_Vals)
    N_N_Seconds_NoCue = length(N_Seconds_NoCue)
    N_mem_slopes = length(mem_slopes)

    delay_prob_correct = zeros(N_N_Object_vals, N_N_Seconds_NoCue, length(mem_slopes))

    for N_Obj_idx = 1:N_N_Object_vals

        # First we'll simulate the model without a retro-cue
    
        N_Objects = N_Object_Vals[N_Obj_idx]
        print(N_Objects)
        N_Seconds = N_Seconds_NoCue[end]
    
        N_TimeSteps_Pre = Int(round(N_Seconds * NT_per_Sec))
        N_TimeSteps_Post = 0
        Relevant_Timepoint = Int.(round.(N_Seconds_NoCue.*NT_per_Sec))

    
        # prob correct is Num Time Steps X Num Objects X Num Mem Slopes
        prob_correct = simulate_task_mult_ms(N_Quanta, N_Objects, epsilon, N_TimeSteps_Pre, N_TimeSteps_Post, N_Trials, simulate_delayed_memory_episode, Relevant_Timepoint; mem_slopes = mem_slopes, cue_reliability = 1, baseline_prob = .5)
    
        delay_prob_correct[N_Obj_idx, : , :] = prob_correct#prob_correct[Relevant_Timepoint, 1, :];

        GC.gc(true)

    end

    return delay_prob_correct

end

# create function to simulate retro-cure conditions
function simulate_retrocue(N_Object_Vals, N_Seconds_Pre_Vals, N_Seconds_Post_Vals, mem_slopes, N_Quanta, epsilon, NT_per_Sec)

    N_N_Object_vals = length(N_Object_Vals)
    N_mem_slopes = length(mem_slopes)
    N_N_Seconds_Pre_Vals = length(N_Seconds_Pre_Vals)
    N_N_Seconds_Post_Vals = length(N_Seconds_Post_Vals)

    # for post cue, always go to the end
    N_Seconds_Post = N_Seconds_Post_Vals[end]
    N_TimeSteps_Post = Int(ceil(N_Seconds_Post * NT_per_Sec))

    retro_cue_prob_correct = zeros(N_N_Object_vals, N_N_Seconds_Pre_Vals, N_N_Seconds_Post_Vals, N_mem_slopes)

    for N_Obj_idx = 1:N_N_Object_vals
        
        N_Objects = N_Object_Vals[N_Obj_idx]
        print(N_Objects)
        
        # loop through all pre-timepoints
        for N_Sec_Pre_idx = 1:length(N_Seconds_Pre_Vals)
            N_Seconds_Pre = N_Seconds_Pre_Vals[N_Sec_Pre_idx]
            N_TimeSteps_Pre = Int(round(N_Seconds_Pre * NT_per_Sec))
            Relevant_Timepoint = Int.(round.((N_Seconds_Pre .+  N_Seconds_Post_Vals).*NT_per_Sec))

            prob_correct = simulate_task_mult_ms(N_Quanta, N_Objects, epsilon, N_TimeSteps_Pre, N_TimeSteps_Post, N_Trials, simulate_retrocue_episode, Relevant_Timepoint; mem_slopes = mem_slopes, cue_reliability = 1, baseline_prob = .5)
                
            retro_cue_prob_correct[N_Obj_idx, N_Sec_Pre_idx, :, :] = prob_correct#[Relevant_Timepoint, 1, :]; # also only care about the first item
            GC.gc(true)

        end
    end

    return retro_cue_prob_correct

end

# list all conditions we want to run
N_Object_Vals = [2, 3, 4, 5, 6, 8, 10];

# Other values are time before cue and time after cue
N_Seconds_Pre_Vals = .25 * 2 .^ (0:5)
N_Seconds_Post_Vals = .25 * 2 .^ (0:5)

# For trials with no retro-cue - start from .25 seconds and multiply by 1.5 8 times
N_Seconds_NoCue = .25 * 2 .^ (0:6)

# store all condition values in a dictionary
condition_vals = Dict("N_Object_Vals" => N_Object_Vals, "N_Seconds_Pre_Vals" => N_Seconds_Pre_Vals, "N_Seconds_Post_Vals" => N_Seconds_Post_Vals, "N_Seconds_NoCue" => N_Seconds_NoCue)

# set parameters for each job... 
eps_vals = collect(1:-.02:.01) # 17

# quanta values
q_vals = collect(2:4:100) #

# NT vals
NT_vals = [800]#[25, 50, 100, 200, 400, 800]

# memory slopes (this won't be looped over)
mem_slopes = [.025, .05, .1, .2, .4, .8, 1.6]

job_EPS = []
job_NQ = []
job_NT = []

for ep in eps_vals
    for q in q_vals
        for nt in NT_vals

            push!(job_EPS, ep)
            push!(job_NQ, q)
            push!(job_NT, nt)

        end
    end
end

n_jobs_total = length(job_NT)

println("N_Jobs_Total: $n_jobs_total")

n_jobs_per_run = 25 # should be about 30 minutes...

n_runs = Int(ceil(n_jobs_total/n_jobs_per_run))

jobs_per_run = reshape(Vector(1:(n_jobs_per_run*n_runs)), (n_jobs_per_run, n_runs))'

these_jobs = jobs_per_run[run_idx,:]

N_Trials = 1000 # consider making this 1000

for this_job_idx in these_jobs

    if (this_job_idx > n_jobs_total)
        break
    end

    local N_Quanta = job_NQ[this_job_idx]
    local epsilon = job_EPS[this_job_idx]
    local NT_per_Second = job_NT[this_job_idx]

    println("Job: $this_job_idx, N_Quanta: $N_Quanta, epsilon: $epsilon, NT_per_Second: $NT_per_Second")

    local file_name = "N_Quanta_$(N_Quanta)_epsilon_$(epsilon)_NT_$(NT_per_Second).jld2"

    # run the delayed memory condition
    local delay_prob_correct = simulate_delayed_memory(N_Object_Vals, N_Seconds_NoCue, mem_slopes, N_Quanta, epsilon, NT_per_Second)
    local full_file_path = joinpath(to_save_folder,"delayed_memory",file_name)    
    jldsave(full_file_path; delay_prob_correct)

    # run the retro-cue condition
    local retro_cue_prob_correct = simulate_retrocue(N_Object_Vals, N_Seconds_Pre_Vals, N_Seconds_Post_Vals, mem_slopes, N_Quanta, epsilon, NT_per_Second)
    local full_file_path = joinpath(to_save_folder,"retrocue",file_name)    
    jldsave(full_file_path; retro_cue_prob_correct)

    local full_file_path = joinpath(to_save_folder,"condition_vals",file_name)
    jldsave(full_file_path; condition_vals)    

    GC.gc(true)

end



