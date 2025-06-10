import ray
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
# Remove problematic TensorBoard imports
import os

# Initialize Ray
ray.init(ignore_reinit_error=True)

def bbc(x, y):
    return 1 - x**2 + 1 - (y - 1)**2

def train_bbc(config):
    try:
        # Get initial values from config
        x = config["x"]
        y = config["y"]
        
        for step in range(60):
            # Evaluate the function with current parameters
            result = bbc(x, y)
            print(f"[Step {step+1}] x={x:.4f}, y={y:.4f}, result={result:.4f}")
            # Report the result to Ray Tune - using 'metrics' dict instead of keyword arguments
            tune.report(metrics={"objective": result, "iteration": step + 1})

    except Exception as e:
        import traceback
        print("Error occurred in train_bbc:")
        traceback.print_exc()
        raise e

pbt = PopulationBasedTraining(
    time_attr="training_iteration",  # Use standard Ray Tune iteration counter
    metric="objective",
    mode="max",
    perturbation_interval=3, 
    hyperparam_mutations={
        "x": lambda: tune.uniform(-3, 3).sample(),
        "y": lambda: tune.uniform(-3, 3).sample(),
    }
)

# Specify a unique experiment name for easier identification in TensorBoard
experiment_name = "pbt_bbc_local"

# Get absolute path to RL2Grid directory
current_dir = os.path.abspath(".")  # Use current directory instead of __file__
results_dir = os.path.join(current_dir, "ray_results")

# Make sure directory exists
os.makedirs(results_dir, exist_ok=True)

analysis = tune.run(
    train_bbc,
    name=experiment_name,
    storage_path=results_dir,  # Using storage_path instead of local_dir
    scheduler=pbt,
    num_samples=16,
    stop={"done": True},  # 
    config={
        "x": tune.uniform(-3, 3),
        "y": tune.uniform(-3, 3),
    },
    resources_per_trial={"cpu": 1},
    verbose=2,
    max_failures=3,
)

# Get and print the best configuration
best_config = analysis.get_best_config(metric="objective", mode="max")
best_trial = analysis.get_best_trial(metric="objective", mode="max")
print("\nBest (x, y):", best_config)
print("Best objective value:", best_trial.last_result["objective"])
print(f"Expected optimum: x=0, y=1, value=2")

# Print TensorBoard instructions with the correct local path
log_dir = os.path.join(results_dir, experiment_name)
print("\n=== TensorBoard Visualization ===")
print(f"TensorBoard logs are saved to: {log_dir}")
print("To view results in TensorBoard, run the following command in your terminal:")
print(f"tensorboard --logdir={log_dir}")
print("\nOr if you're in a Jupyter notebook, you can run:")
print("# Install TensorBoard if needed")
print("# !pip install tensorboard")
print("from tensorboard import notebook")
print(f"notebook.start('--logdir={log_dir}')")

# Additional note about Ray and TensorBoard
print("\nNote: Ray Tune automatically logs metrics to TensorBoard format")
print("even without special TensorBoard loggers or callbacks.")