import time
import logging
import time

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  
        result = func(*args, **kwargs)
        end_time = time.time()  
        print(f"Execution time for {func.__name__}: {end_time - start_time:.2f} seconds")
        return result
    return wrapper




def setup_logging(log_level=logging.INFO):
    """
    Set up logging for the application.
    """
    logging.basicConfig(
        level=log_level,  
        format="%(asctime)s - %(levelname)s - %(message)s",  
    )
    return logging.getLogger(__name__)
