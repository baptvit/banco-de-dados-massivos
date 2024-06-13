import time
import psutil
import logging


def all_in_one_profile(func, logger):
    """Decorator that measures execution time, memory usage, and CPU usage (may be platform-specific)."""

    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        start_cpu_usage = psutil.cpu_percent()

        result = func(*args, **kwargs)

        end_time = time.perf_counter()
        end_cpu_usage = psutil.cpu_percent()

        elapsed_time = end_time - start_time
        cpu_usage_change = end_cpu_usage - start_cpu_usage

        logger.info(f"{func.__name__} took {elapsed_time} seconds to execute")
        # logging.info(f"{func.__name__} peak memory usage: {memory_usage:.2f} MB")
        logger.info(
            f"{func.__name__} caused CPU usage to increase by {cpu_usage_change}%"
        )
        return result

    return wrapper
