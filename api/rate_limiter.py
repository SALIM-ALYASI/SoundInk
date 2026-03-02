import time
from typing import Dict, Tuple
from fastapi import Request, HTTPException

# Simple memory storage: IP -> (timestamp_of_first_request, request_count)
# For production, Redis should be used.
_RATE_LIMITS: Dict[str, Tuple[float, int]] = {}

def check_rate_limit(request: Request, max_requests: int = 15, time_window_sec: int = 60):
    """
    Enforces a simple sliding window rate limit per IP address.
    Returns True if allowed, raises 429 otherwise.
    """
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    
    if ip not in _RATE_LIMITS:
        _RATE_LIMITS[ip] = (now, 1)
        return True
        
    start_time, count = _RATE_LIMITS[ip]
    
    if now - start_time > time_window_sec:
        # Reset window
        _RATE_LIMITS[ip] = (now, 1)
        return True
        
    if count >= max_requests:
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded. Try again in {int(time_window_sec - (now - start_time))} seconds.")
        
    _RATE_LIMITS[ip] = (start_time, count + 1)
    return True
