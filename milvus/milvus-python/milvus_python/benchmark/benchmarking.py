import time
import psutil

def benchmark_script(script_func):
  """
  Executes a script function and gathers benchmarking information.

  Args:
      script_func: The function representing the script to be benchmarked.

  Returns:
      A dictionary containing benchmarking information (time, memory usage, CPU usage).
  """
  memory_before = psutil.Process().memory_info().rss  # Resident memory before
  cpu_usage_before = psutil.cpu_percent()  # CPU usage before

  start_time = time.perf_counter()
  script_func()
  end_time = time.perf_counter()

  memory_after = psutil.Process().memory_info().rss  # Resident memory after
  cpu_usage_after = psutil.cpu_percent()  # CPU usage after

  execution_time = end_time - start_time
  memory_used = memory_after - memory_before
  cpu_usage_increase = cpu_usage_after - cpu_usage_before

  return {
      "time": execution_time,
      "memory_usage": memory_used,
      "cpu_usage": cpu_usage_increase,
  }

# Example usage
def your_script_function():
  # Your script logic here
  pass

results = benchmark_script(your_script_function)
print(f"Time taken: {results['time']:.4f} seconds")
print(f"Memory used: {results['memory_usage']:,} bytes")
print(f"CPU usage increase: {results['cpu_usage']:.2f}%")
