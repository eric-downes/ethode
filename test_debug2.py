from ethode.controller import ControllerConfig
import traceback

try:
    config = ControllerConfig(
        kp="0.2 / day",
        ki="0.2 / day / 7 day",
        kd="0.0 hour",
        tau="1 week",
        noise_band=("1 milliUSD", "3 milliUSD"),
        output_min="-100 USD",
        output_max="100 USD",
        rate_limit="10 USD/hour",
    )

    print("Config created successfully")
    
    runtime = config.to_runtime(check_units=False)
    print("Runtime created successfully")
    
    from ethode.validation import validate_controller_dimensions
    dimensions = validate_controller_dimensions(runtime)
    print(f"Validation succeeded: {dimensions}")
    
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    traceback.print_exc()
