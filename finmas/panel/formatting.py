from bokeh.models.widgets.tables import DateFormatter, HTMLTemplateFormatter, NumberFormatter

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
    disabled=True,
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
    sorters=[{"field": "date", "dir": "desc"}],
    show_index=False,
    disabled=True,
)

news_config = dict(
    page_size=15,
    pagination="local",
    # formatters={"link": {"type": "link", "target": "_blank"}},
    formatters={
        "link": HTMLTemplateFormatter(
            template=(
                '<a href="<%= link %>" target="_blank"><i class="fas fa-external-link"></i></a>'
            )
        ),
    },
    header_filters={
        "title": {"type": "input", "func": "like", "placeholder": "Keyword"},
        "published": {"type": "input", "func": "like", "placeholder": "YYYY-MM-DD"},
        "num_symbols": {"type": "number", "func": "<=", "placeholder": "Max amount"},
    },
    hidden_columns=["id", "summary", "content", "markdown_content", "text", "symbols"],
    # max_width=1000,
    layout="fit_data_fill",
    sizing_mode="stretch_width",
    # buttons={"url": '<i class="fas fa-external-link"></i>'},
    # widths={"title": "60%", "published": "15%", "num_symbols": "15%"},
    show_index=False,
    disabled=True,
)

llm_models_config = dict(show_index=False, disabled=True)
