# AI_Data_Centers_waiting_queue_heuristic
waiting queue, max 1000 GPU, limited supply, heuristic, 80 jobs as lambda, not linear programming
Problem with v2.py 
It assigns all jobs in state (t,s) to run at the same k frequency, u_k is only 0 or 1. 
LP will likely change that to make u_k continuos, that is what the technical note wants, I believe.
