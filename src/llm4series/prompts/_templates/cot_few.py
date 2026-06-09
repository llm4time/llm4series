# autopep8: off

COT_FEW = """\
[OBJECTIVE]
Predict the next {forecast_horizon} values based on the historical series ({input_len} periods).

[STATISTICAL CONTEXT]
{statistics}

[REASONING GUIDELINES]
- Trend: Identify the overall direction (increasing, decreasing, stable) and the trend strength.
- Seasonality: Patterns that repeat at regular intervals (e.g., daily, weekly, monthly).
- Outliers: Possible outliers or abrupt changes.
- Cycles: Not seasonal long-term patterns.
- Noise reduction: Apply a technique to reduce noise when necessary.
- Consistency with the provided descriptive statistics (mean, median, etc.).
- Adjustment for data frequency and contextual events (holidays, promotions, etc.).

[EXAMPLES]
{forecast_examples}

[INPUT]"""
