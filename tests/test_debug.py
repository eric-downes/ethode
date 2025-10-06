from ethode.controller import ControllerConfig

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

report = config.validate_units(verbose=True)
print(f"\nSuccess: {report.success}")
print(f"Errors: {report.errors}")
