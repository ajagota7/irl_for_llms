import subprocess
import time

def run_experiment(n_runs=10):
    base_command = [
        "PYTHONPATH=.", "python", "src/irl.py",
        "experiment=exp_70m",
        "experiment.learn_rm=EleutherAI/pythia-70m",
        "experiment.learn_rm_size=",
        "experiment.training.n_epochs=5",
        "experiment.training.sample_size=100"
    ]
    
    for run in range(n_runs):
        print(f"\n{'='*50}")
        print(f"Starting Run {run + 1}/{n_runs}")
        print(f"{'='*50}\n")
        
        # Run the command
        process = subprocess.run(" ".join(base_command), shell=True)
        
        # Check if the run was successful
        if process.returncode != 0:
            print(f"Run {run + 1} failed with return code {process.returncode}")
        
        # Small delay between runs to ensure clean separation
        time.sleep(5)
        
        print(f"\nCompleted Run {run + 1}/{n_runs}")

if __name__ == "__main__":
    run_experiment(10)
