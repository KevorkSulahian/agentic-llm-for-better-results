import os
from crewai import Agent, Task, Crew, Process

# from crewai import Agent, Task, Crew, Process, LLM
from tools.sentiment_analysis import ticker_sentiment_analysis
from tools.technical_analysis import yf_tech_analysis
from tools.analyze_data import yf_fundamental_analysis

# from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain_openai import ChatOpenAI
# from dotenv import load_dotenv

# Environment Variables


# Model Selection
def initialize_llm(model_option, api_key=None, groq_api_key=None):
    if model_option == "ollama":
        return ChatOpenAI(
            openai_api_base="http://localhost:11434/v1",
            openai_api_key="ollama",
            model_name="ollama/llama3.1:8b-instruct-q6_K",
        )
        # return LLM(model= f"ollama/llama3.1:8b-instruct-q6_K", base_url="http://localhost:11434")
    elif model_option == "OpenAI GPT-4o":
        return ChatOpenAI(openai_api_key=api_key, model="gpt-4o", temperature=0.1)
    elif model_option == "OpenAI GPT-4o Mini":
        return ChatOpenAI(openai_api_key=api_key, model="gpt-4o-mini", temperature=0.1)

    else:
        raise ValueError("Invalid model option selected")


def create_crew(stock_symbol, model_option, API_KEY=None):
    llm = initialize_llm(model_option)
    # Tools Initialization
    sentiment_analysis_tool = ticker_sentiment_analysis

    yf_tech_tool = yf_tech_analysis
    yf_fundamental_tool = yf_fundamental_analysis

    # Agents Definitions
    researcher = Agent(
        role="Senior Stock Market Researcher",
        goal="Gather and analyze comprehensive data about {stock_symbol}",
        verbose=True,
        memory=True,
        backstory="With a Ph.D. in Financial Economics and 15 years of experience in equity research, you're known for your meticulous data collection and insightful analysis.",
        tools=[sentiment_analysis_tool],
        llm=llm,
    )

    technical_analyst = Agent(
        role="Expert Technical Analyst",
        goal="Perform an in-depth technical analysis on {stock_symbol}",
        verbose=True,
        memory=True,
        backstory="As a Chartered Market Technician (CMT) with 15 years of experience, you have a keen eye for chart patterns and market trends.",
        tools=[yf_tech_tool],
        llm=llm,
    )

    fundamental_analyst = Agent(
        role="Senior Fundamental Analyst",
        goal="Conduct a comprehensive fundamental analysis of {stock_symbol}",
        verbose=True,
        memory=True,
        backstory="With a CFA charter and 15 years of experience in value investing, you dissect financial statements and identify key value drivers.",
        tools=[yf_fundamental_tool],
        llm=llm,
    )

    reporter = Agent(
        role="Chief Investment Strategist",
        goal="Synthesize all analyses to create a definitive investment report on {stock_symbol}",
        verbose=True,
        memory=True,
        backstory="As a seasoned investment strategist with 20 years of experience, you weave complex financial data into compelling investment narratives.",
        tools=[
            yf_fundamental_tool,
            yf_tech_tool,
            # YahooFinanceNewsTool(),
            sentiment_analysis_tool,
        ],  # reddit_tool, serper_tool,
        llm=llm,
    )

    # Task Definitions
    # research_task = Task(
    #     description=(
    #         "Conduct research on {stock_symbol}. Your analysis should include:\n"
    #         "1. Current stock price and historical performance (5 years).\n"
    #         "2. Key financial metrics (P/E, EPS growth, revenue growth, margins).\n"
    #         "3. Recent news and press releases (1 month).\n"
    #         "4. Analyst ratings and price targets (min 3 analysts).\n"
    #         "5. sentiment analysis from yfinance (100 posts).\n"
    #         "6. Major institutional holders and recent changes.\n"
    #         "7. Competitive landscape and {stock_symbol}'s market share.\n"
    #         "Use reputable financial websites for data."
    #     ),
    #     expected_output="A detailed 150-word research report with data sources and brief analysis.",
    #     agent=researcher,
    # )

    research_task = Task(
        description=(
            "Conduct research on {stock_symbol}. Your analysis should include:\n"
            "The results of tool sentiment analysis on {stock_symbol}.\n"
            "The tool's function name is ticker_sentiment_analysis and you can call it using {stock_symbol} and a keyword that corresponds to {stock_symbol}.\n"
            "And this will return the overall sentiment, number of articles, and sentiment counts. \n"
            "Use the ticker_sentiment_analysis tool for data."
        ),
        expected_output="A detailed 150-word research report on the current sentiment of the ticker in the market.",
        agent=researcher,
    )

    technical_analysis_task = Task(
        description=(
            "Perform technical analysis on {stock_symbol}. Include:\n"
            "1. 50-day and 200-day moving averages (1 year).\n"
            "2. Key support and resistance levels (3 each).\n"
            "3. RSI and MACD indicators.\n"
            "4. Volume analysis (3 months).\n"
            "5. Significant chart patterns (6 months).\n"
            "6. Fibonacci retracement levels.\n"
            "7. Comparison with sector's average.\n"
            "Use the yf_tech_analysis tool for data."
        ),
        expected_output="A 100-word technical analysis report with buy/sell/hold signals and annotated charts.",
        agent=technical_analyst,
    )

    fundamental_analysis_task = Task(
        description=(
            "Conduct fundamental analysis of {stock_symbol}. Include:\n"
            "1. Review last 3 years of financial statements.\n"
            "2. Key ratios (P/E, P/B, P/S, PEG, Debt-to-Equity, etc.).\n"
            "3. Comparison with main competitors and industry averages.\n"
            "4. Revenue and earnings growth trends.\n"
            "5. Management effectiveness (ROE, capital allocation).\n"
            "6. Competitive advantages and market position.\n"
            "7. Growth catalysts and risks (2-3 years).\n"
            "8. DCF valuation model with assumptions.\n"
            "Use yf_fundamental_analysis tool for data."
        ),
        expected_output="A 100-word fundamental analysis report with buy/hold/sell recommendation and key metrics summary.",
        agent=fundamental_analyst,
    )

    report_task = Task(
        description=(
            "Create an investment report on {stock_symbol}. Include:\n"
            "1. Executive Summary: Investment recommendation.\n"
            "2. Company Snapshot: Key facts.\n"
            "3. Financial Highlights: Top metrics and peer comparison.\n"
            "4. Technical Analysis: Key findings.\n"
            "5. Fundamental Analysis: Top strengths and concerns.\n"
            "6. Risk and Opportunity: Major risk and growth catalyst.\n"
            "7. Sentiment: Key takeaway from sentiment analysis, that includes number of articles used and the overal score.\n"
            "8. Investment Thesis: Bull and bear cases.\n"
            "9. Price Target: 12-month forecast.\n"
        ),
        expected_output="A 600-word investment report with clear sections, key insights.",
        agent=reporter,
    )

    # Crew Definition and Kickoff for Result
    crew = Crew(
        agents=[researcher, technical_analyst, fundamental_analyst, reporter],
        tasks=[research_task, technical_analysis_task, fundamental_analysis_task, report_task],
        process=Process.sequential,
        cache=True,
        planning=True,
    )

    result = crew.kickoff(inputs={"stock_symbol": stock_symbol})

    os.makedirs("./crew_results", exist_ok=True)
    file_path = f"./crew_results/crew_result_{stock_symbol}.md"

    result_str = f"Token Usage: {result.token_usage} \n\n" + str(result)
    with open(file_path, "w") as file:
        file.write(result_str)

    return file_path
