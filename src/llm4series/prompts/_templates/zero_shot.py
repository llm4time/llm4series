# autopep8: off

ZERO_SHOT = """\
[OBJECTIVE]
Predict the next {forecast_horizon} values based on the historical series ({input_len} periods).

[STATISTICAL CONTEXT]
{statistics}

[INPUT]"""
