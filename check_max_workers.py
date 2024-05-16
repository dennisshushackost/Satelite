import concurrent.futures
import os
import psutil

# Get the number of physical CPU cores
num_cores = os.cpu_count() // 2  # Assuming hyperthreading, divide by 2 to get physical cores

# Set number of workers for CPU-bound tasks
cpu_bound_workers = num_cores

# Set number of workers for I/O-bound tasks (2-4 times the number of cores)
io_bound_workers = min(num_cores * 4, num_cores + 50)  # Cap it to avoid excessive threading

# Ensure we don't exceed RAM usage
ram_usage_per_worker = 500  # Example RAM usage per worker in MB (adjust based on your task)
available_ram = psutil.virtual_memory().available // (1024 * 1024)  # Convert bytes to MB
max_workers_based_on_ram = available_ram // ram_usage_per_worker

# Final number of workers
cpu_bound_workers = min(cpu_bound_workers, max_workers_based_on_ram)
io_bound_workers = min(io_bound_workers, max_workers_based_on_ram)

print(f"CPU-bound workers: {cpu_bound_workers}")
print(f"I/O-bound workers: {io_bound_workers}")

