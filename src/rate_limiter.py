import time
from collections import deque
from threading import Lock


class ThreadSafeRateLimiter:
  """
  A thread-safe rate limiter to manage API calls across multiple threads.

  This limiter ensures that the number of requests made within a given time
  period does not exceed a specified limit.
  """

  def __init__(self, max_requests: int, period_seconds: int):
    """
    Initializes the rate limiter.

    Args:
        max_requests: The maximum number of requests allowed in the period.
        period_seconds: The time period in seconds.
    """
    self.max_requests = max_requests
    self.period_seconds = period_seconds
    self.request_timestamps = deque()
    self.lock = Lock()

  def acquire(self):
    """
    Acquires a permit from the rate limiter, blocking if necessary.

    This method will block until a request can be made without exceeding the
    rate limit.
    """
    while True:
      time_to_wait = 0
      with self.lock:
        current_time = time.monotonic()
        # Remove timestamps older than the period
        while (
          self.request_timestamps
          and self.request_timestamps[0] <= current_time - self.period_seconds
        ):
          self.request_timestamps.popleft()

        if len(self.request_timestamps) < self.max_requests:
          self.request_timestamps.append(current_time)
          break
        else:
          # Calculate the time to wait for the oldest request to expire
          time_to_wait = (
            self.request_timestamps[0] + self.period_seconds - current_time
          )

      if time_to_wait > 0:
        print(f"Rate limit exceeded. Waiting for {time_to_wait:.2f} seconds...")
        time.sleep(time_to_wait)
