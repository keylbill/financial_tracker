def get_options_data(ticker: str) -> pd.DataFrame:
    """
    Fetch options data for a given ticker symbol.
    
    Args:
        ticker (str): The stock ticker symbol
        
    Returns:
        pd.DataFrame: DataFrame containing options data with columns for both calls and puts.
                     Returns empty DataFrame if data cannot be fetched.
    """
    try:
        stock = yf.Ticker(ticker)
        if not stock:
            logger.error(f"Could not initialize ticker {ticker}")
            return pd.DataFrame()
            
        expirations = stock.options
        if not expirations:
            logger.error(f"No options data available for {ticker}")
            return pd.DataFrame()
            
        # Get options data for the nearest expiration date
        nearest_expiry = expirations[0]
        try:
            # Fetch both calls and puts
            calls = stock.option_chain(nearest_expiry).calls
            puts = stock.option_chain(nearest_expiry).puts
            
            # Combine the data
            calls['option_type'] = 'call'
            puts['option_type'] = 'put'
            options_data = pd.concat([calls, puts], ignore_index=True)
            
            return options_data
            
        except Exception as e:
            logger.error(f"Error fetching options chain for {ticker} expiry {nearest_expiry}: {e}")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error fetching options data for ticker {ticker}: {e}")
        return pd.DataFrame() 