import datetime as dt

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def get_ta_plot_figure(df: pd.DataFrame) -> go.Figure:
    """Returns a Plotly figure with a candlestick chart and RSI."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        # subplot_titles=("OHLC", "RSI"),
        row_heights=[0.7, 0.3],
    )

    fig.add_trace(
        go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="OHLC",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(x=df["date"], y=df["sma_20"], mode="lines", name="SMA 20"), row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=df["date"], y=df["sma_50"], mode="lines", name="SMA 50"), row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=df["date"], y=df["rsi_14"], mode="lines", name="RSI 14"), row=2, col=1
    )
    fig.add_shape(
        type="line",
        x0=df["date"].min(),
        x1=df["date"].max(),
        y0=70,
        y1=70,
        line=dict(dash="dash"),
        row=2,
        col=1,
    )
    fig.add_shape(
        type="line",
        x0=df["date"].min(),
        x1=df["date"].max(),
        y0=30,
        y1=30,
        line=dict(dash="dash"),
        row=2,
        col=1,
    )
    fig.update_yaxes(range=[10, 90], row=2, col=1)

    fig.update_layout(
        autosize=True,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="",
        yaxis=dict(side="left"),
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=0),
    )
    fig.update_xaxes(
        autorange=False,
        range=[df["date"].max() - dt.timedelta(days=365 * 2), df["date"].max()],
        rangeselector=dict(
            buttons=list(
                [
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(count=2, label="2y", step="year", stepmode="backward"),
                    dict(count=3, label="3y", step="year", stepmode="backward"),
                    dict(step="all"),
                ]
            ),
        ),
        rangeslider=dict(visible=False),
        type="date",
        row=1,
        col=1,
    )
    fig.update_xaxes(
        rangeslider=dict(visible=True),
        type="date",
        row=2,
        col=1,
    )

    return fig


def get_income_statment_plot_figure(df: pd.DataFrame) -> go.Figure:
    """Returns a Plotly figure with a grouped bar chart for the income statement data."""
    fig = go.Figure(
        data=[
            go.Bar(x=df.index, y=df["totalRevenue"], name="Total Revenue"),
            go.Bar(x=df.index, y=df["grossProfit"], name="Gross Profit"),
            go.Bar(x=df.index, y=df["operatingExpenses"], name="Operating Expenses"),
            go.Bar(x=df.index, y=df["netIncome"], name="Net Income"),
        ],
        layout=go.Layout(barmode="group"),
    )

    fig.update_layout(
        autosize=True,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="",
        yaxis=dict(title="USD", side="right"),
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=0),
    )
    return fig
