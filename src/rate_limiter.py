import time
from collections import deque
from threading import Lock


class ThreadSafeRateLimiter:
  """
  A thread-safe rate limiter to manage API calls across multiple threads.
  This limiter ensures that the number of requests made within a given time
  period does not exceed a specified limit.
  """

  def __init__(
    self,
    max_requests: int,
    period_seconds: int,
    log_interval_requests: int = 100,
    log_interval_seconds: float = 5.0,
  ):
    """
    Initializes the rate limiter.

    Args:
        max_requests: The maximum number of requests allowed in the period.
        period_seconds: The time period in seconds.
        log_interval_requests: How often (in requests) to print a status update.
        log_interval_seconds: Minimum time (in seconds) between "Rate limit exceeded" messages.
    """
    if max_requests < 1 or period_seconds < 1:
      raise ValueError("Max requests and period must be greater than 0.")
    self.max_requests = max_requests
    self.period_seconds = period_seconds
    self.log_interval_requests = log_interval_requests
    self.log_interval_seconds = (
      log_interval_seconds  # For throttling wait messages
    )

    self.request_timestamps = deque()
    self.lock = Lock()

    # --- ADDED: Timestamp for the last "waiting" log message ---
    self._last_wait_log_time = 0

  def acquire(self):
    """
    Acquires a permit from the rate limiter, blocking if necessary.
    """
    while True:
      time_to_wait = 0
      with self.lock:
        current_time = time.monotonic()

        while (
          self.request_timestamps
          and self.request_timestamps[0] <= current_time - self.period_seconds
        ):
          self.request_timestamps.popleft()

        if len(self.request_timestamps) < self.max_requests:
          self.request_timestamps.append(current_time)

          if len(self.request_timestamps) % self.log_interval_requests == 0:
            print(
              f"Rate Limiter Status: {len(self.request_timestamps)} requests in last {self.period_seconds}s. (Capacity: {self.max_requests})"
            )

          break
        else:
          time_to_wait = (
            self.request_timestamps[0] + self.period_seconds - current_time
          )

      if time_to_wait > 0:
        now = time.monotonic()
        if now - self._last_wait_log_time > self.log_interval_seconds:
          print(
            f"Rate limit reached. Waiting for ~{time_to_wait:.2f} seconds..."
          )
          self._last_wait_log_time = now  # Update the log time

        time.sleep(time_to_wait)
