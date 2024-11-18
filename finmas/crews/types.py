from finmas.crews import (
    CombinedCrew,
    MarketDataCrew,
    NewsAnalysisCrew,
    SECFilingSectionsCrew,
    SECFilingCrew,
)

CrewType = NewsAnalysisCrew | SECFilingSectionsCrew | MarketDataCrew | CombinedCrew | SECFilingCrew
