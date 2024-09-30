from bokeh.models.widgets.tables import DateFormatter, NumberFormatter

income_statement_config = dict(
    formatters={
        "fiscalDateEnding": DateFormatter(format="%Y-%m-%d"),
        "totalRevenue": NumberFormatter(format="0.0a"),
        "operatingExpenses": NumberFormatter(format="0.0a"),
        "grossProfit": NumberFormatter(format="0.0a"),
        "netIncome": NumberFormatter(format="0.0a"),
        "netProfitMargin": NumberFormatter(format="0.0%"),
    },
    text_align="center",
    page_size=5,
    pagination="local",
)

ohlcv_config = dict(
    formatters={
        "date": DateFormatter(format="%Y-%m-%d"),
        "close": NumberFormatter(format="0,0.00"),
        "volume": NumberFormatter(format="0,0"),
    },
    hidden_columns=["open", "high", "low"],
    text_align="center",
    page_size=5,
    pagination="local",
    show_index=False,
    sorters=[{"field": "date", "dir": "desc"}],
)
