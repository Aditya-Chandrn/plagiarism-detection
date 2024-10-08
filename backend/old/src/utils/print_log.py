import traceback

def log_success(message):
  print(f"✅ {message}", flush=True)
  
def log_error(message, error):
  print(f"❌ {message} :- {error}", flush=True)
  print(traceback.format_exc(), flush=True)
  
def log_failure(message):
  print(f"⚠️ {message}", flush=True)