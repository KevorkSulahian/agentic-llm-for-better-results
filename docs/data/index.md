# Data Sources

This is a list of the data sources that have been integrated in this app.
Those data sources that require an API Key are marked in the table.
Then it is necessary to set those as a environment variable which can
be done by compting the `.env.template` file in the root folder and
creating a `.env` file with the API keys.

| API Key                         | Name               | Type             | Description                             |
| ------------------------------- | ------------------ | ---------------- | --------------------------------------- |
|                                 | Yahoo Finance      | Price data       |                                         |
|                                 | SEC / Edgar        | Filings          |                                         |
|                                 | Yahoo Finance News | News             | Free access to the latest News via RSS  |
| <i class="fa-solid fa-key"></i> | Benzinga           | News             | Requires registration of Alpaca account |
| <i class="fa-solid fa-key"></i> | Alpha Vantage      | Fundamental data | Limited to 25 calls per day.            |
