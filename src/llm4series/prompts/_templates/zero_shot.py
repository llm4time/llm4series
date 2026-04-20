# autopep8: off

ZERO_SHOT = """\
Objective:
Predict the next {forecast_horizon} values based on the historical series ({input_len} periods).

Statistical Context (to guide the forecast):
{statistics}

Steps:
1. Analyze the series step by step (internally; do not include this in the final output).
2. Generate the forecast for the next {forecast_horizon} periods.
3. Format the output exactly as in the example, with values inside <out>.

Example (strict reference format):
<out>
{output_example}
</out>

Input (series data for forecast):"""
