import datetime as dt

import hvplot.pandas  # noqa: F401
import pandas as pd
import panel as pn
import param
import plotly.graph_objects as go
import yfinance as yf
from panel.viewable import Viewable

from finmanalysis.panel.constants import defaults

pn.extension("tabulator", "plotly", template="fast")


class FinMAnalysis(pn.viewable.Viewer):
    ticker_select = pn.widgets.Select(name="Ticker", width=100)
    start_picker = pn.widgets.DatetimeInput(
        name="Start",
        format="%Y-%m-%d",
        value=dt.date.fromisoformat(defaults["start_date"]),
        width=100,
    )
    end_picker = pn.widgets.DatetimeInput(
        name="End",
        format="%Y-%m-%d",
        value=dt.date.today(),
        width=100,
    )
    update_counter = pn.widgets.IntInput(value=0)

    fetch_data_btn = pn.widgets.Button(name="Fetch data", button_type="primary")
    ohlcv_df = param.DataFrame(pd.DataFrame())

    def __init__(self, **params) -> None:
        super().__init__(**params)

        self.ticker_select.value = defaults["tickerid"]
        self.ticker_select.options = defaults["tickerids"]

        self.fetch_data_btn.on_click(self.fetch_yf_data)
        self.fetch_yf_data(None)

    def fetch_yf_data(self, event) -> None:
        with self.fetch_data_btn.param.update(loading=True):
            df = yf.download(
                self.ticker_select.value,
                start=self.start_picker.value,
                end=self.end_picker.value,
                interval="1d",
                auto_adjust=True,
                threads=True,
                progress=False,
            )
            df = df.reset_index()
            df.columns = df.columns.str.lower()
            self.ohlcv_df = df
            self.update_counter.value += 1

    def get_tickerid_plot(self, *args, **kwargs):
        if self.ohlcv_df.empty:
            return None
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=self.ohlcv_df["date"],
                    open=self.ohlcv_df["open"],
                    high=self.ohlcv_df["high"],
                    low=self.ohlcv_df["low"],
                    close=self.ohlcv_df["close"],
                )
            ]
        )
        fig.update_layout(
            autosize=True,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            title=self.ticker_select.value,
        )
        return fig

    def __panel__(self) -> Viewable:
        return pn.Row(
            pn.WidgetBox(
                self.ticker_select,
                pn.Row(self.start_picker, self.end_picker),
                self.fetch_data_btn,
                width=300,
            ),
            pn.Column(
                pn.Card(
                    pn.pane.Plotly(
                        pn.bind(self.get_tickerid_plot, update_counter=self.update_counter)
                    ),
                    width=800,
                    height=600,
                    margin=10,
                    title="Price Plot",
                ),
            ),
        )


if pn.state.served:
    FinMAnalysis().servable(title="FinMAnalysis")
