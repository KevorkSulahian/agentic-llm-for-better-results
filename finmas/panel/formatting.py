from bokeh.models.widgets.tables import DateFormatter, HTMLTemplateFormatter, NumberFormatter

INCOME_STATEMENT_COLS_MAP = {
    "totalRevenue": "Total Revenue",
    "grossProfit": "Gross Profit",
    "operatingExpenses": "OpEx",
    "netIncome": "Net Income",
    "netProfitMargin": "Net Profit Margin (%)",
}

income_statement_config = dict(
    titles=INCOME_STATEMENT_COLS_MAP,
    formatters={
        "date": DateFormatter(format="%Y-%m-%d"),
        "totalRevenue": NumberFormatter(format="0.0a"),
        "operatingExpenses": NumberFormatter(format="0.0a"),
        "grossProfit": NumberFormatter(format="0.0a"),
        "netIncome": NumberFormatter(format="0.0a"),
        "netProfitMargin": NumberFormatter(format="0.0"),
    },
    text_align="center",
    page_size=5,
    pagination="local",
    disabled=True,
)

ta_config = dict(
    titles={
        "date": "Date",
        "close": "Close",
        "sma_20": "SMA 20",
        "sma_50": "SMA 50",
        "sma_trend": "SMA Trend",
        "rsi_14": "RSI 14",
        "bb_pband": "BB %",
        "ret_1w": "1W Ret. (%)",
    },
    formatters={
        "date": DateFormatter(format="%Y-%m-%d"),
        "close": NumberFormatter(format="0,0.00"),
        "sma_20": NumberFormatter(format="0,0.00"),
        "sma_50": NumberFormatter(format="0,0.00"),
        "rsi_14": NumberFormatter(format="0"),
        "bb_pband": NumberFormatter(format="0"),
        "ret_1w": NumberFormatter(format="0.0%"),
    },
    hidden_columns=["open", "high", "low"],
    text_align="center",
    page_size=5,
    pagination="local",
    sorters=[{"field": "date", "dir": "desc"}],
    show_index=False,
    disabled=True,
)


def ta_styler(styler):
    styler.background_gradient(axis=None, vmin=20, vmax=80, cmap="Spectral_r", subset=["rsi_14"])
    styler.background_gradient(axis=None, vmin=0, vmax=100, cmap="Spectral_r", subset=["bb_pband"])
    styler.map(trend_style, subset=["sma_trend"])
    styler.map(float_color_style, subset=["ret_1w"])
    return styler


def trend_style(v):
    if v == "Up":
        return "color:green;"
    elif v == "Down":
        return "color:red;"
    else:
        return "color:black;"


def float_color_style(v):
    return "color:red;" if v < 0 else "color:green;"


news_config = dict(
    titles={
        "title": "Title",
        "published": "Published",
        "author": "Author",
        "num_symbols": "Num Symbols",
        "link": "Link",
    },
    page_size=20,
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
        "author": {"type": "input", "func": "like", "placeholder": "Author"},
        "published": {"type": "input", "func": "like", "placeholder": "YYYY-MM-DD"},
        "num_symbols": {"type": "number", "func": ">=", "placeholder": "Min amount"},
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

embedding_models_config = dict(
    formatters={
        "link": HTMLTemplateFormatter(
            template=(
                '<a href="<%= link %>" target="_blank"><i class="fas fa-external-link"></i></a>'
            )
        )
    },
    show_index=False,
    disabled=True,
)

tickers_config = dict(
    page_size=30,
    pagination="local",
    titles={
        "ticker": "Ticker",
        "name": "Name",
        "market_cap": "Market Cap",
        "sector": "Sector",
        "industry_group": "Industry Group",
        "industry": "Industry",
        "market": "Market",
        "website": "Website",
    },
    formatters={
        "website": HTMLTemplateFormatter(
            template=(
                '<a href="<%= website %>" target="_blank"><i class="fas fa-external-link"></i></a>'
            )
        ),
    },
    header_filters={
        "ticker": {"type": "input", "func": "like", "placeholder": "Ticker"},
        "name": {"type": "input", "func": "like", "placeholder": "Name"},
        "sector": {"type": "input", "func": "like", "placeholder": "Sector"},
        "industry_group": {"type": "input", "func": "like", "placeholder": "Industry Group"},
        "industry": {"type": "input", "func": "like", "placeholder": "Industry"},
        "market": {"type": "input", "func": "like", "placeholder": "Market"},
        "market_cap": {"type": "input", "func": "like", "placeholder": "Large"},
    },
    show_index=False,
    disabled=True,
)

sec_filings_config = dict(
    page_size=10,
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
        "form": {"type": "input", "func": "like", "placeholder": "10-Q"},
        "filing_date": {"type": "input", "func": "like", "placeholder": "YYYY-MM-DD"},
        "reportDate": {"type": "input", "func": "like", "placeholder": "YYYY-MM-DD"},
    },
    hidden_columns=["accession_number"],
    # max_width=1000,
    layout="fit_data_fill",
    show_index=False,
    disabled=True,
)
