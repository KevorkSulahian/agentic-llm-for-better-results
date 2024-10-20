from sec_tool import SECTools
from edgar import set_identity

# Set identity for SEC API access
set_identity("John Doe john.doe@example.com")


def test_sec_tools():
    sec_tool = SECTools()

    # Test 10-K filing
    stock_ticker = "NVDA"
    query = "What was the gross profit last year?"

    # Search in 10-K
    try:
        print(f"Searching 10-K filing for {stock_ticker}...")
        result_10k = sec_tool.search_10k(stock_ticker, query)
        print("Search Result from 10-K:")
        print(result_10k)
    except Exception as e:
        print(f"Error searching 10-K: {e}")

    # 10-Q filing for the company
    try:
        print("=" * 10)
        print(f"\n\nSearching 10-Q filing for {stock_ticker}...")
        result_10q = sec_tool.search_10q(stock_ticker, query)
        print("Search Result from 10-Q:")
        print(result_10q)
    except Exception as e:
        print(f"Error searching 10-Q: {e}")


if __name__ == "__main__":
    test_sec_tools()
