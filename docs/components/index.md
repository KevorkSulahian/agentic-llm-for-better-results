# Components

## Architecture

The app is developed in a component-based fashion where dedicated components are each responsible for a separate part of the system. This makes it easier to further extend the system if needed with additional functionality within each component. Each component has its respective folder in the codebase.
The main architecture of the app consists of 4 main components:

- **panel**: Component that handles the user interface, all user interactions, and data visualization.
  The component configures the system with default settings and lets the user further customize the system.
- **crews:** Component that is responsible for setting up and running dedicated
  Multi-agent systems (crews) that either focus on a specific data source or multiple data sources.
  Tools that the Agents use to perform their tasks are defined here.
- **data:** This component focuses on fetching and processing data from sources such as SEC filings, Benzinga News,
  Alpha Vantage API, and Yahoo Finance.
- **utils:** The objective of this component is to gather helper functions that are used across the
  different components in a common place.

```mermaid
{% include 'diagrams/finmas_architecture.mmd' %}
```
