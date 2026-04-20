# autopep8: off

COT = """\
Objective:
Predict the next {forecast_horizon} values based on the historical series ({input_len} periods).

Statistical Context (to guide the forecast):
{statistics}

Reasoning Instructions:
Before generating the forecast, analyze the historical series step by step, considering:
- Trend: Identify the overall direction (increasing, decreasing, stable) and the trend strength.
- Seasonality: Patterns that repeat at regular intervals (e.g., daily, weekly, monthly).
- Outliers: Possible outliers or abrupt changes.
- Cycles: Not seasonal long-term patterns.
- Noise reduction: Apply a technique to reduce noise when necessary.
- Consistency with the provided descriptive statistics (mean, median, etc.).
- Adjustment for data frequency and contextual events (holidays, promotions, etc.).

Steps:
1. Analyze the series step by step (internally; do not include this in the final output).
2. Generate the forecast for the next {forecast_horizon} periods.
3. Format the output exactly as in the example, with values inside <out>.

Example (strict reference format):
<out>
{output_example}
</out>

Input (series data for forecast):"""
