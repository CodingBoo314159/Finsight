import pandas as pd
import numpy as np

def rule_based_categorisation(df, description_column='description'):
    # lowercase everything so we don't miss any matches
    df['clean_desc'] = df[description_column].astype(str).str.lower()
    
    # setup empty columns for our results
    df['Category'] = np.nan
    df['Rule_Reason'] = np.nan
    
    # our 9 categories from the graph plus local SA places
    merchant_rules = {
        'Groceries': ['checkers', 'pick n pay', 'woolworths', 'spar', 'shoprite'],
        'Money Sent': ['ewallet', 'cashsend', 'instant money', 'send money'],
        'Banking': ['bank charges', 'monthly fee', 'service fee', 'atm withdrawal'],
        'Food': ['mr d food', 'uber eats', 'kfc', 'nandos', 'mcdonalds', 'debonairs'],
        'Transport': ['engen', 'sasol', 'bp', 'shell', 'gautrain', 'uber', 'bolt'],
        'Shopping': ['takealot', 'mr price', 'cotton on', 'edgars', 'superbalist'],
        'Entertainment': ['ster kinekor', 'computicket', 'netflix', 'showmax', 'spotify'],
        'Utilities': ['city of tshwane', 'eskom', 'telkom', 'vodacom', 'mtn'],
        'Other': [] # leave blank, XGBoost will handle the tricky ones
    }
    
    # loop through the dictionary to check for matches
    for category, merchants in merchant_rules.items():
        for merchant in merchants:
            # find rows with the merchant name that aren't sorted yet
            match_mask = df['clean_desc'].str.contains(merchant, na=False) & df['Category'].isna()
            
            # assign the category and leave a note on why it matched
            df.loc[match_mask, 'Category'] = category
            df.loc[match_mask, 'Rule_Reason'] = f"Rule match: '{merchant}'"
            
    # split the data: sorted ones stay here, the leftovers go to the ML model
    categorised_data = df[df['Category'].notna()].drop(columns=['clean_desc'])
    uncategorised_data = df[df['Category'].isna()].drop(columns=['clean_desc'])
    
    return categorised_data, uncategorised_data