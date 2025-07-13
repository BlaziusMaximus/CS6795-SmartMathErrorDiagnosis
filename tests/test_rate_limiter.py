import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.rate_limiter import ThreadSafeRateLimiter


def test_rate_limiter_initialization():
  """Tests that the rate limiter initializes with the correct properties."""
  limiter = ThreadSafeRateLimiter(max_requests=10, period_seconds=60)
  assert limiter.max_requests == 10
  assert limiter.period_seconds == 60
  assert len(limiter.request_timestamps) == 0


def acquire_permit(limiter: ThreadSafeRateLimiter):
  """A simple target function for threads to call."""
  limiter.acquire()
  return True


def test_rate_limiter_throttles_requests():
  """
  Tests the core logic of the rate limiter by spawning more threads than
  the request limit and measuring the total execution time.
  """
  # Arrange
  num_requests = 5
  max_requests_per_period = 3
  period_seconds = 1  # Use a short period for a fast test

  limiter = ThreadSafeRateLimiter(
    max_requests=max_requests_per_period, period_seconds=period_seconds
  )

  start_time = time.monotonic()

  # Act
  with ThreadPoolExecutor(max_workers=num_requests) as executor:
    futures = [
      executor.submit(acquire_permit, limiter) for _ in range(num_requests)
    ]

    # Wait for all futures to complete
    results = [future.result() for future in as_completed(futures)]

  end_time = time.monotonic()

  total_duration = end_time - start_time

  # Assert
  # The first 3 requests should pass instantly. The 4th request must wait
  # for the first request's timestamp to expire, which is after 1 second.
  # Therefore, the total time must be greater than the period.
  assert len(results) == num_requests
  assert all(results)  # Check that all threads eventually succeeded
  assert total_duration >= period_seconds


def test_rate_limiter_timestamps_are_cleared():
  """
  Tests that old timestamps are correctly cleared from the deque.
  """
  # Arrange
  limiter = ThreadSafeRateLimiter(max_requests=2, period_seconds=1)

  # Act
  limiter.acquire()  # t = 0.0
  time.sleep(1.1)  # Wait for the first timestamp to become old
  limiter.acquire()  # t = 1.1, the old timestamp should be cleared here

  # Assert
  # If the old timestamp was cleared, there should only be one timestamp left.
  assert len(limiter.request_timestamps) == 1
