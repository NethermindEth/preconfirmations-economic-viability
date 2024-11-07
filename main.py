import pandas as pd
from web3.exceptions import BadFunctionCallOutput
from web3 import Web3
import matplotlib.pyplot as plt
import requests
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.optimize import curve_fit
import codecs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


"""" 
Plot Style Conditioning and Misc Utils
"""
# Set the style globally at the beginning
plt.rc('font', family='serif', size=11)

def apply_classic_style(ax):
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    ax.grid(False)
    ax.set_facecolor('white')  # Ensure the background of the chart area is white

def block_to_eth_price(block_number):
    df = pd.read_parquet('Data/Data/Brontes/Blocks.parquet')
    df = df[['block_number', 'eth_price']]
    return df[df['block_number'] == block_number]['eth_price'].values[0]

def get_price_eth_for_token(token_address, block_number):
    infura_url = 'https://eth-mainnet.rpc.nethermind.io/rpc'
    web3 = Web3(Web3.HTTPProvider(infura_url))
    
    # Convert addresses to checksum format
    token_address = web3.to_checksum_address(token_address)
    UNISWAP_V3_FACTORY_ADDRESS = web3.to_checksum_address('0x1F98431c8aD98523631AE4a59f267346ea31F984')
    UNISWAP_V2_FACTORY_ADDRESS = web3.to_checksum_address('0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f')
    WETH_ADDRESS = web3.to_checksum_address('0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2')  # WETH address
    thegraph_v3_url = 'https://gateway.thegraph.com/api/fe6d283860da4baf0f3028c07be36708/subgraphs/id/5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV'
    thegraph_v2_url = 'https://gateway.thegraph.com/api/fe6d283860da4baf0f3028c07be36708/subgraphs/id/EYCKATKGBKLWvSfwvBjzfCBmGwYNdVkduYXVivCsLRFu'

    if token_address == WETH_ADDRESS:
        return 1  # WETH price in WETH is 1

    # ABI definitions
    ERC20_ABI = '[{"constant":true,"inputs":[],"name":"symbol","outputs":[{"name":"","type":"string"}],"payable":false,"stateMutability":"view","type":"function"},{"constant":true,"inputs":[],"name":"decimals","outputs":[{"name":"","type":"uint8"}],"payable":false,"stateMutability":"view","type":"function"},{"constant":true,"inputs":[],"name":"name","outputs":[{"name":"","type":"string"}],"payable":false,"stateMutability":"view","type":"function"}]'
    UNISWAP_POOL_ABI = '[{"constant":true,"inputs":[],"name":"token0","outputs":[{"name":"","type":"address"}],"payable":false,"stateMutability":"view","type":"function"},{"constant":true,"inputs":[],"name":"token1","outputs":[{"name":"","type":"address"}],"payable":false,"stateMutability":"view","type":"function"},{"constant":true,"inputs":[],"name":"slot0","outputs":[{"name":"sqrtPriceX96","type":"uint160"}],"payable":false,"stateMutability":"view","type":"function"}]'
    UNISWAP_V3_FACTORY_ABI = '[{"inputs":[{"internalType":"address","name":"","type":"address"},{"internalType":"address","name":"","type":"address"},{"internalType":"uint24","name":"","type":"uint24"}],"name":"getPool","outputs":[{"internalType":"address","name":"","type":"address"}],"payable":false,"stateMutability":"view","type":"function"}]'
    UNISWAP_V2_FACTORY_ABI = '[{"inputs":[{"internalType":"address","name":"","type":"address"},{"internalType":"address","name":"","type":"address"}],"name":"getPair","outputs":[{"internalType":"address","name":"","type":"address"}],"stateMutability":"view","type":"function"}]'

    # Get the token's decimal information
    token_contract = web3.eth.contract(address=token_address, abi=ERC20_ABI)
    token_decimals = token_contract.functions.decimals().call()

    # Get the pool address
    uniswap_v3_factory_contract = web3.eth.contract(address=UNISWAP_V3_FACTORY_ADDRESS, abi=UNISWAP_V3_FACTORY_ABI)
    pool_address_v3 = uniswap_v3_factory_contract.functions.getPool(token_address, WETH_ADDRESS, 3000).call()

    uniswap_v2_factory_contract = web3.eth.contract(address=UNISWAP_V2_FACTORY_ADDRESS, abi=UNISWAP_V2_FACTORY_ABI)
    pool_address_v2 = uniswap_v2_factory_contract.functions.getPair(token_address, WETH_ADDRESS).call()
    print(f"V3 Pool Address: {pool_address_v3}, V2 Pair Address: {pool_address_v2}")

    #print(f"V3 Pool Address: {pool_address_v3}, V2 Pair Address: {pool_address_v2}")

    if(pool_address_v3 != '0x0000000000000000000000000000000000000000'):

        queryV3 = f"""
        {{
        pool(id: "{pool_address_v3.lower()}", block: {{number: {block_number}}}) {{
            tick
            token0 {{
            symbol
            id
            decimals
            }}
            token1 {{
            symbol
            id
            decimals
            }}
            feeTier
            sqrtPrice
            liquidity
            token1Price
            token0Price
        }}
        }}
        """

        response = requests.post(thegraph_v3_url, json={'query': queryV3})
        data = response.json()
        print(data)

        try:
            token0 = data['data']['pool']['token0']['id']
            token1 = data['data']['pool']['token1']['id']
        except:
            return None
        
        if token0 == token_address:
            price = data['data']['pool']['token1Price']
        else:
            price = data['data']['pool']['token0Price']

    else:
        queryV2 = f"""
        {{
        pair(id: "{pool_address_v2.lower()}", block: {{number: {block_number}}}) {{
            id
            token0 {{
            id
            symbol
            decimals
            }}
            token1 {{
            id
            symbol
            decimals
            }}
            token1Price
            token0Price
            reserve0
            reserve1
        }}
        }}
        """
        response = requests.post(thegraph_v2_url, json={'query': queryV2})
        data = response.json()
        print(data)

        # Determine which token in the pool is the one of interest
        try:
            token0 = data['data']['pair']['token0']['id']
            token1 = data['data']['pair']['token1']['id']
        except:
            return None

        if token0 == token_address:
            price = data['data']['pair']['token1Price']
        else:
            price = data['data']['pair']['token0Price']


    price = float(price)
    return price

def get_block_for_tx_hash(tx_hash):
    infura_url = 'https://eth-mainnet.rpc.nethermind.io/rpc'
    web3 = Web3(Web3.HTTPProvider(infura_url))
    try:
        tx_receipt = web3.eth.get_transaction_receipt(tx_hash)
        return tx_receipt.blockNumber
    except BadFunctionCallOutput:
        return None

def decode_byte_sequence(byte_input, byteorder='big'):
    """
    Decodes a byte sequence into an integer.

    Parameters:
    - byte_input (str or bytes): The byte sequence to decode.
    - byteorder (str): Byte order used to represent the integer. 
                       'big' for Big-Endian, 'little' for Little-Endian.

    Returns:
    - int or np.nan: The decoded integer, or NaN if decoding fails.
    """
    try:
        if isinstance(byte_input, str):
            # Convert string with escape sequences to bytes
            byte_sequence = codecs.decode(byte_input, 'unicode_escape')
        elif isinstance(byte_input, bytes):
            byte_sequence = byte_input
        else:
            # Unsupported type
            return np.nan
        
        # Convert bytes to integer
        number = int.from_bytes(byte_sequence, byteorder=byteorder)
        return number
    except Exception as e:
        # Optionally, log the error or pass
        # For now, we'll return NaN for any decoding issues
        return np.nan

def decode_byte_sequence(x):
    return int.from_bytes(x, 'big') if isinstance(x, bytes) else x



""""
Breakdown of Proposer Tyles by different type of reward. 
"""

def classification_breakdown():
    
    # Loading Block Data
    df_block = pd.read_parquet('Data/Brontes/blocks.parquet')
    df_block['total_priority_fee'] = df_block['total_priority_fee'].apply(lambda x: decode_byte_sequence(x)) / 10**18
    df_block['total_bribe'] = df_block['total_bribe'].apply(lambda x: decode_byte_sequence(x)) / 10**18
    df_block['total_mev_priority_fee_paid'] = df_block['total_mev_priority_fee_paid'].apply(lambda x: decode_byte_sequence(x)) / 10**18
    df_block['proposer_mev_reward'] = df_block['proposer_mev_reward'].apply(lambda x: decode_byte_sequence(x)) / 10**18
    df_block['total_proposer_reward'] = df_block['proposer_mev_reward'] + df_block['total_priority_fee'] + df_block['total_bribe']
    df_block['percentage_of_total_builder_payment'] = df_block['proposer_mev_reward'] / df_block['total_proposer_reward']
    print(df_block['percentage_of_total_builder_payment'].mean())
    
    # Loading Header Data With Updated CEX-DEX Tagging. 
    df2 = pd.read_parquet('Data/Brontes/updated_header.parquet')
    
    # Grouping and restructuring MEV data
    df2 = df2.groupby(['block_number', 'mev_type'])['bribe_usd'].sum().reset_index()
    df2 = df2.pivot(index='block_number', columns='mev_type', values='bribe_usd').reset_index()
    df2.fillna(0, inplace=True)  # Fill any missing values with 0 to avoid NaN issues
    
    # Reorganizing MEV types
    df2['Cex Dex'] = df2.get('CexDexQuotes') + df2.get('CexDexRfq') + df2.get('CexDexTrades')  # Combine different CexDex types
    df2['JIT Sandwich'] = df2.get('Jit') + df2.get('JitSandwich')  # Combine different JIT Sandwich types
    df2.drop(columns=['Jit', 'JitSandwich'], errors='ignore', inplace=True)
    df2.drop(columns=['CexDexQuotes', 'CexDexRfq', 'CexDexTrades'], errors='ignore', inplace=True)
    df2.rename(columns={'AtomicArb': 'Atomic Arbitrage'}, inplace=True)  # Rename for clarity

    # Merging the eth price data to the main dataframe on the nearest time
    df_main = pd.merge_asof(df_block.sort_values('block_number'), df2.sort_values('block_number'), on='block_number', direction='nearest')

    # Renaming some columns for clarity
    df_main.rename(columns={'proposer_mev_reward': 'Proposers Coinbase Transfer', 'SearcherTx': 'Complex'}, inplace=True)

    # Changing values from USD to ETH using avg_price
    # Divide all MEV-related columns by the ETH price to convert USD to ETH
    mev_columns = ['Atomic Arbitrage', 'JIT Sandwich', 'Cex Dex', 'Liquidation', 'Sandwich', 'Complex']
    for mev_type in mev_columns:
        df_main[mev_type] = df_main[mev_type] / df_main['eth_price']
    
    # Calculate total MEV sum by summing the specified columns
    df_main['total_mev_sum'] = df_main[mev_columns].sum(axis=1)

    # Adjust each MEV type with added coinbase transfer proportionally
    for column in mev_columns:
        added_transfer = df_main['Proposers Coinbase Transfer'] * (df_main[column] / df_main['total_mev_sum'])  # Proportional adjustment
        df_main[column + '_added_coinbase_transfer'] = added_transfer  # Store added transfer separately
        df_main[column] += added_transfer  # Add it to the original column

    # Recalculate the total MEV sum after adjusting
    df_main['total_mev_sum'] = df_main[mev_columns].sum(axis=1)

    # Calculate remainder by subtracting MEV sum from total proposer reward
    df_main['Remainder'] = df_main['total_proposer_reward'] - df_main['total_mev_sum']
    df_main = df_main.where(df_main['Remainder'] >= 0).dropna()  # Drop rows where remainder is negative
    df_analysis = df_main[['Remainder']] # Create dataframe for analysis
    df_analysis = pd.concat([df_analysis, df_main[mev_columns]], axis=1) # Add MEV columns to analysis dataframe

    # Ensure MEV columns are numeric
    df_main[mev_columns] = df_main[mev_columns].apply(pd.to_numeric, errors='coerce')

    # Generate descriptive statistics
    df_stats = df_main[mev_columns + ['total_mev_sum', 'total_proposer_reward']].describe()

    # Finding average bribe amount per MEV type and plotting the pie chart
    df_analysis_avg = df_analysis.mean()  # Calculate average values for analysis

    # Manually filter out non-numeric values from the Series
    numeric_columns = df_analysis_avg[df_analysis_avg.apply(lambda x: isinstance(x, (float, int)))]

    labels = numeric_columns.index  # Labels for the pie chart
    sizes = numeric_columns.values  # Values (sizes) for the pie chart

    # Sum of all average constituents
    total_sum = sizes.sum()

    # Calculate percentages
    percentages = (sizes / total_sum) * 100

    # Printing average bribe and proposer reward statistics
    print('Sum Of Average Proposer Reward Constituents:', total_sum)  # Sum of all average constituents
    print('Average Proposer Reward:', df_main['total_proposer_reward'].mean())  # Average proposer reward

    # Print each MEV type's percentage of the total
    for label, size, percentage in zip(labels, sizes, percentages):
        print(f'{label}: {size:.4f} ETH, {percentage:.2f}% of total')

    # Plotting pie chart for MEV breakdown
    colors = plt.get_cmap('tab20')(range(len(labels)))  # Use 'tab20' colormap
    plt.figure(figsize=(8, 8))  # Create a figure for the pie chart
    plt.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',  # Show percentages
        startangle=140,  # Rotate start of pie chart
        colors=colors,
        wedgeprops={'edgecolor': 'white'}  # Add white edges to wedges for better visibility
    )
    plt.axis('equal')  # Ensure pie chart is a circle
    plt.show()  # Display the pie chart

    # Continue with the rest of your function...
    return df_analysis_avg


""""
----------------
Subblock Scaling
----------------
"""

""""
Derivation of Sandwich Scaling Factor. 
"""

def sandwich_processor():
    df = pd.read_parquet('Data/Brontes/Sandwich.parquet')

    def process_row(row, index):
        try:
            block_number = get_block_for_tx_hash(row['backrun_tx_hash'])
            gas_used_backrun = pd.to_numeric(row['backrun_gas_details']['gas_used'])
            gas_used_frontrun = pd.to_numeric(row['frontrun_gas_details'][0]['gas_used'])
            base_fee = pd.to_numeric(row['backrun_gas_details']['effective_gas_price']) - pd.to_numeric(row['backrun_gas_details']['priority_fee']) 
            frontrun_coinbase_transfer = pd.to_numeric(row['frontrun_gas_details'][0]['coinbase_transfer']) if row['frontrun_gas_details'][0]['coinbase_transfer'] is not None else 0
            backrun_coinbase_transfer = pd.to_numeric(row['backrun_gas_details']['coinbase_transfer']) if row['backrun_gas_details']['coinbase_transfer'] is not None else 0

            priority_fee_backrun = pd.to_numeric(row['backrun_gas_details']['priority_fee'])
            priority_fee_frontrun = pd.to_numeric(row['frontrun_gas_details'][0]['priority_fee']) 

            return {
                'block_number': block_number,
                'gas_used_backrun': gas_used_backrun,
                'gas_used_frontrun': gas_used_frontrun,
                'base_fee': base_fee,
                'coinbase_transfer_frontrun': frontrun_coinbase_transfer,
                'coinbase_transfer_backrun': backrun_coinbase_transfer,
                'priority_fee_backrun': priority_fee_backrun,
                'priority_fee_frontrun': priority_fee_frontrun
            }

        except Exception as e:
            print(f"Error processing row {index}: {e}")
            return None
        
    # Use ThreadPoolExecutor for multithreading
    with ThreadPoolExecutor(max_workers=20) as executor:
        # Submit tasks and get futures
        futures = [executor.submit(process_row, row, index) for index, row in df.iterrows()]
        
        # Process results as they complete
        results = []
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing rows"):
            result = future.result()
            if result is not None:
                results.append(result)

    # Create a new DataFrame from the results
    new_df = pd.DataFrame(results)
    
    # Save to CSV
    new_df.to_csv('Data/Brontes/SandwichExaminerResults.csv', index=False)
    print(new_df)

def sandwich_subblock_algo(df_analysis_avg):

    # Load data
    df = pd.read_csv('Data/Brontes/SandwichExaminerResults.csv')
    df = df.groupby('block_number').sum()

    # Calculate averages
    average_gas_used_frontrun = df['gas_used_frontrun'].mean()
    average_gas_used_backrun = df['gas_used_backrun'].mean()
    average_base_fee = df['base_fee'].mean()
    df['total_base_fee'] = (df['gas_used_frontrun'] + df['gas_used_backrun']) * df['base_fee'] / 10**18
    average_total_base_fee_frontrun = average_base_fee * average_gas_used_frontrun / 10**18
    average_total_base_fee_backrun = average_base_fee * average_gas_used_backrun / 10**18
    average_total_base_fee_eth = average_total_base_fee_frontrun + average_total_base_fee_backrun

    average_proposer_rewards = df_analysis_avg['Sandwich'] 
    print(f'Average Proposer Reward Sanwich: {average_proposer_rewards} ETH')
    print(f'Average Base Fee Sanwich: {average_total_base_fee_eth} ETH')
    print(f'Total Revenue Sandwich: {average_proposer_rewards + average_total_base_fee_eth} ETH')

    # New sub-block accumulation algorithm
    def proposer_rewards_with_accumulation(s, average_proposer_rewards, average_total_base_fee_eth):
        total_reward = 0
        tally = 0  # Accumulated sub-blocks
        
        total_rev = average_proposer_rewards + average_total_base_fee_eth

        for sub_slot in range(s + 1):

            tally += 1

            # Calculate potential reward and gas cost
            potential_reward = total_rev / ( ( (s + 1) / tally )) 
            gas_cost = average_total_base_fee_eth

            # Check if a sandwich is profitable
            if potential_reward > gas_cost:
                total_reward += potential_reward - gas_cost
                tally = 0  # Reset accumulated volume

        return total_reward

    # Calculate the proposer rewards for each sub-block setting including the 0 case
    range_given = range(0, 40)  # Include 0 in the range
    print(average_proposer_rewards, average_total_base_fee_eth)
    proposer_rewards_list = [proposer_rewards_with_accumulation(s, average_proposer_rewards, average_total_base_fee_eth) for s in range_given]
    
    # Plotting splits against rewards
    plt.plot(range_given, proposer_rewards_list, '-o', label='Proposer Rewards')

    # Exponential decay fitting function
    def exponential_decay(x, a, b):
        return a * np.exp(-b * x)

    # Fit the exponential decay model to the data
    params, _ = curve_fit(exponential_decay, range_given, proposer_rewards_list, p0=(1, 0.1))
    a, b = params

    # Print the equation
    print(f'Fit Equation: Sandwich proposer_rewards = {a:.4f} * exp(-{b:.4f} * s)')

    # Plot the exponential decay model
    y_pred = exponential_decay(np.array(range_given), a, b)
    plt.plot(range_given, y_pred, label='Exponential Decay Fit', color='grey')
    plt.xlabel('Number of Splits')
    plt.ylabel('Proposer Rewards (ETH)')
    plt.title(f'Sandwich Proposer Rewards vs. Number of Splits')
    plt.legend()
    plt.show()

""""
Derivation of JIT Sandwich Sandwich Scaling Factor. 
"""
def jit_processor():
    df = pd.read_parquet('./Data/Data/Brontes/JIT_Sandwich.parquet')

    def process_row(row, index):
        try:
            block_number = get_block_for_tx_hash(row['backrun_tx_hash'])
            gas_used_backrun = pd.to_numeric(row['backrun_gas_details']['gas_used'])
            gas_used_frontrun = pd.to_numeric(row['frontrun_gas_details'][0]['gas_used'])
            base_fee = pd.to_numeric(row['backrun_gas_details']['effective_gas_price']) - pd.to_numeric(row['backrun_gas_details']['priority_fee']) 
            frontrun_coinbase_transfer = pd.to_numeric(row['frontrun_gas_details'][0]['coinbase_transfer']) if row['frontrun_gas_details'][0]['coinbase_transfer'] is not None else 0
            backrun_coinbase_transfer = pd.to_numeric(row['backrun_gas_details']['coinbase_transfer']) if row['backrun_gas_details']['coinbase_transfer'] is not None else 0
            priority_fee_backrun = pd.to_numeric(row['backrun_gas_details']['priority_fee'])
            priority_fee_frontrun = pd.to_numeric(row['frontrun_gas_details'][0]['priority_fee']) 

            return {
                'block_number': block_number,
                'gas_used_backrun': gas_used_backrun,
                'gas_used_frontrun': gas_used_frontrun,
                'base_fee': base_fee,
                'coinbase_transfer_frontrun': frontrun_coinbase_transfer,
                'coinbase_transfer_backrun': backrun_coinbase_transfer,
                'priority_fee_backrun': priority_fee_backrun,
                'priority_fee_frontrun': priority_fee_frontrun
            }

        except Exception as e:
            print(f"Error processing row {index}: {e}")
            return None
        
    # Use ThreadPoolExecutor for multithreading
    with ThreadPoolExecutor(max_workers=20) as executor:
        # Submit tasks and get futures
        futures = [executor.submit(process_row, row, index) for index, row in df.iterrows()]
        
        # Process results as they complete
        results = []
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing rows"):
            result = future.result()
            if result is not None:
                results.append(result)

    # Create a new DataFrame from the results
    new_df = pd.DataFrame(results)
    
    # Save to CSV
    new_df.to_csv('Data/Brontes/JITSandwichSandwichExaminerResults.csv', index=False)

def jit_subblock_algo(df_analysis_avg):

    # Load data
    df = pd.read_csv('Data/Brontes/JITSandwichExaminerResults.csv')
    df = df.groupby('block_number').sum()

    # Calculate averages
    average_gas_used_frontrun = df['gas_used_frontrun'].mean()
    average_gas_used_backrun = df['gas_used_backrun'].mean()
    average_base_fee = df['base_fee'].mean()
    average_total_base_fee_frontrun = average_base_fee * average_gas_used_frontrun / 10**18
    average_total_base_fee_backrun = average_base_fee * average_gas_used_backrun / 10**18
    average_total_base_fee_eth = average_total_base_fee_frontrun + average_total_base_fee_backrun

    average_proposer_rewards = df_analysis_avg['JIT Sandwich'] 
    print(f'Average Proposer Reward JIT: {average_proposer_rewards} ETH')
    print(f'Average Base Fee JIT: {average_total_base_fee_eth} ETH')
    print(f'Total Revenue JIT: {average_proposer_rewards + average_total_base_fee_eth} ETH')

    # New sub-block accumulation algorithm
    def proposer_rewards_with_accumulation(s, average_proposer_rewards, average_total_base_fee_eth):
        total_reward = 0
        tally = 0  # Accumulated reward
        total_rev = average_proposer_rewards + average_total_base_fee_eth
        for sub_slot in range(s + 1):

            tally += 1

            # Calculate potential reward and gas cost
            potential_reward = total_rev / ( ( (s + 1) / tally ))  # From OLS regression, the reward as a function of volume 
            gas_cost = average_total_base_fee_eth

            # Check if a sandwich is profitable
            if potential_reward > gas_cost:
                total_reward += potential_reward - gas_cost
                tally = 0  # Reset accumulated volume

        return total_reward


    # Calculate the proposer rewards for each sub-block setting including the 0 case
    range_given = range(0, 40)  # Include 0 in the range
    proposer_rewards_list = [proposer_rewards_with_accumulation(s, average_proposer_rewards, average_total_base_fee_eth) for s in range_given]

    # Plotting splits against rewards
    plt.plot(range_given, proposer_rewards_list, '-o', label='Proposer Rewards')

    # Exponential decay fitting function
    def exponential_decay(x, a, b):
        return a * np.exp(-b * x)

    # Fit the exponential decay model to the data
    params, _ = curve_fit(exponential_decay, range_given, proposer_rewards_list, p0=(1, 0.1))
    a, b = params

    # Print the equation
    print(f'Fit Equation: JIT Sandwich proposer_rewards = {a:.4f} * exp(-{b:.4f} * s)')

    # Plot the exponential decay model
    y_pred = exponential_decay(np.array(range_given), a, b)
    plt.plot(range_given, y_pred, label='Exponential Decay Fit', color='grey')
    plt.xlabel('Number of Splits')
    plt.ylabel('Proposer Rewards (ETH)')
    plt.title(f'JIT Sandwich Proposer Rewards vs. Number of Splits')
    plt.legend()
    plt.show()

""""
Derivation of Atomic Scaling Factor. 
"""
def atomic_processor():
    df = pd.read_parquet('Data/Brontes/Atomic.parquet')
    print(df.iloc[0])

    def process_row(index, row):
        block_number = get_block_for_tx_hash(row['tx_hash'])
        base_fee = pd.to_numeric(row['gas_details']['effective_gas_price']) - pd.to_numeric(row['gas_details']['priority_fee']) 
        gas_used = pd.to_numeric(row['gas_details']['gas_used'])
        priority_fee = pd.to_numeric(row['gas_details']['priority_fee'])
        coinbase_transfer = pd.to_numeric(row['gas_details']['coinbase_transfer']) if row['gas_details']['coinbase_transfer'] is not None else 0
        return index, base_fee, gas_used, priority_fee, coinbase_transfer, block_number

    # Using ThreadPoolExecutor for multithreading
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(process_row, i, df.iloc[i]): i for i in range(len(df))}
        
        # Using tqdm to display a progress bar
        for future in tqdm(futures, total=len(df), desc="Processing Rows"):
            index, base_fee, gas_used, priority_fee, coinbase_transfer, block_number = future.result()
            df.loc[index, 'base_fee'] = base_fee
            df.loc[index, 'gas_used'] = gas_used
            df.loc[index, 'priority_fee'] = priority_fee
            df.loc[index, 'coinbase_transfer'] = coinbase_transfer
            df.loc[index, 'block_number'] = block_number

    df.to_csv('Data/Brontes/AtomicExaminerResults.csv', index=False)

def atomic_subblock_algo(df_analysis_avg):
    # Load data
    df = pd.read_csv('Data/Brontes/AtomicExaminerResults.csv')
    df = df.groupby('block_number').sum()
    
    # Calculate averages
    df['total_base_fee'] = df['gas_used'] * df['base_fee'] / 10**18
    average_total_base_fee_eth = df['total_base_fee'].mean()
    average_proposer_rewards = df_analysis_avg['Atomic Arbitrage']

    print(f'Average Proposer Reward Atomic: {average_proposer_rewards} ETH')
    print(f'Average Base Fee Atomic: {average_total_base_fee_eth} ETH')
    print(f'Total Revenue Atomic: {average_proposer_rewards + average_total_base_fee_eth} ETH')

    # New sub-block accumulation algorithm
    def proposer_rewards_with_accumulation(s, average_proposer_rewards, average_total_base_fee_eth):
        total_reward = 0
        tally = 0  
        
        total_rev = average_proposer_rewards + average_total_base_fee_eth

        for sub_slot in range(s + 1):

            tally += 1

            # Calculate potential reward and gas cost
            potential_reward = total_rev / ( ( (s + 1) / tally )) 
            gas_cost = average_total_base_fee_eth

            # Check if a sandwich is profitable
            if potential_reward > gas_cost:
                #print('Executed! Added to proposer Reward: ' , potential_reward - gas_cost)
                total_reward += potential_reward - gas_cost
                tally = 0  # Reset accumulated volume

        return total_reward


    # Calculate the proposer rewards for each sub-block setting including the 0 case
    range_given = range(0, 40)  # Include 0 in the range
    proposer_rewards_list = [proposer_rewards_with_accumulation(s, average_proposer_rewards, average_total_base_fee_eth) for s in range_given]

    # Plotting splits against rewards
    plt.plot(range_given, proposer_rewards_list, '-o', label='Proposer Rewards')

    # Exponential decay fitting function
    def exponential_decay(x, a, b):
        return a * np.exp(-b * x)

    # Fit the exponential decay model to the data
    params, _ = curve_fit(exponential_decay, range_given, proposer_rewards_list, p0=(1, 0.1))
    a, b = params

    # Print the equation
    print(f'Fit Equation: Atomic Sandwich proposer_rewards = {a:.4f} * exp(-{b:.4f} * s)')

    # Plot the exponential decay model
    y_pred = exponential_decay(np.array(range_given), a, b)
    plt.plot(range_given, y_pred, label='Exponential Decay Fit', color='grey')
    plt.xlabel('Number of Splits')
    plt.ylabel('Proposer Rewards (ETH)')
    plt.title(f'Atomic Sandwich Proposer Rewards vs. Number of Splits')
    plt.legend()
    plt.show()

""""
Derivation of Cex-Dex Scaling Factor. 
"""

def cex_dex_algo(df_analysis_avg):
    df = pd.read_csv('Data/updated_fb_non_atomics.csv')
    df = df.groupby('block_number').sum()

    # Merging base fee onto it
    df_base_fee = pd.read_csv('./Data/base_fee_fb_time.csv')
    df_base_fee.rename(columns={'number': 'block_number'}, inplace=True)

    df = pd.merge(df, df_base_fee, on='block_number', how='left')
    df.drop(columns=['Unnamed: 0.1', 'Unnamed: 0_x', 'Unnamed: 0_y'], inplace=True)
    df['total_base_fee'] = df['gas_used_x'] * df['base_fee'] / 10**18
    #print(df)

    # Retrieving average proposer rewards
    average_proposer_rewards = df_analysis_avg['Cex Dex']
    average_total_base_fee_eth = df['total_base_fee'].mean()
    print(f'Average Proposer Reward: {average_proposer_rewards} ETH')
    print(f'Average Total Base Fee: {average_total_base_fee_eth} ETH')
    print(f'Average Revenue: {average_proposer_rewards + average_total_base_fee_eth} ETH')

    # New sub-block accumulation algorithm
    def proposer_rewards_with_accumulation(s, average_proposer_rewards, average_total_base_fee_eth):
        total_reward = 0
        tally = 0  # Accumulated reward
        
        total_rev = average_proposer_rewards + average_total_base_fee_eth
        for sub_slot in range(s + 1):
            tally += 1

            # Calculate potential reward and gas cost
            potential_reward = total_rev / ( ( (s + 1) / tally ) **  (3/2))  
            gas_cost = average_total_base_fee_eth

            # Check if CEX-DEX is profitable
            if potential_reward > gas_cost:
                total_reward += potential_reward - gas_cost
                tally = 0  # Reset tally

        return total_reward
    
    # Calculate the proposer rewards for each sub-block setting including the 0 case
    range_given = range(0, 40)
    proposer_rewards_list = [proposer_rewards_with_accumulation(s, average_proposer_rewards, average_total_base_fee_eth) for s in range_given]

    # Plotting splits against rewards
    plt.plot(range_given, proposer_rewards_list, '-o', label='Proposer Rewards')

    # Exponential decay fitting function
    def exponential_decay(x, a, b):
        return a * np.exp(-b * x)

    # Fit the exponential decay model to the data
    params, _ = curve_fit(exponential_decay, range_given, proposer_rewards_list, p0=(1, 0.))
    a, b = params

    # Print the equation
    print(f'Fit CEX-DEX Equation: proposer_rewards = {a:.4f} * exp(-{b:.4f} * s)')

    # Plot the exponential decay model
    y_pred = exponential_decay(np.array(range_given), a, b)
    plt.plot(range_given, y_pred, label='Exponential Decay Fit', color='grey')
    plt.xlabel('Number of Splits')
    plt.ylabel('Proposer Rewards (ETH)')
    plt.title(f'CEX DEX Proposer Rewards vs. Number of Splits')
    plt.legend()
    plt.show()

""""
Derivation of Searcher Scaling Factor. 
"""

def searcher_processing():
    df = pd.read_parquet('./Data/Data/Brontes/Searcher.parquet')
    print(df)
    import time
    def process_row(index, row):
        block_number = get_block_for_tx_hash(row['tx_hash'])
        base_fee = pd.to_numeric(row['gas_details']['effective_gas_price']) - pd.to_numeric(row['gas_details']['priority_fee']) 
        gas_used = pd.to_numeric(row['gas_details']['gas_used'])
        priority_fee = pd.to_numeric(row['gas_details']['priority_fee'])
        coinbase_transfer = pd.to_numeric(row['gas_details']['coinbase_transfer']) if row['gas_details']['coinbase_transfer'] is not None else 0
        time.sleep(0.001)
        return index, base_fee, gas_used, priority_fee, coinbase_transfer, block_number

    # Using ThreadPoolExecutor for multithreading
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(process_row, i, df.iloc[i]): i for i in range(len(df))}
        
        # Using tqdm to display a progress bar
        for future in tqdm(futures, total=len(df), desc="Processing Rows"):
            index, base_fee, gas_used, priority_fee, coinbase_transfer, block_number = future.result()
            df.loc[index, 'base_fee'] = base_fee
            df.loc[index, 'gas_used'] = gas_used
            df.loc[index, 'priority_fee'] = priority_fee
            df.loc[index, 'coinbase_transfer'] = coinbase_transfer
            df.loc[index, 'block_number'] = block_number

    df.to_csv('Data/Brontes/SearcherExaminerResults.csv', index=False)

def searcher_subblock_algo(df_analysis_avg):

    # Load data
    df = pd.read_csv('Data/Brontes/SearcherExaminerResults.csv')
    df = df.groupby('block_number').sum()

    # Calculate averages
    average_gas_used = df['gas_used'].mean()
    average_base_fee = df['base_fee'].mean()
    average_total_base_fee_eth = average_base_fee * average_gas_used / 10**18

    # Average Proposer rewards
    average_proposer_rewards = df_analysis_avg['Complex']

    print(f'Average Proposer Reward: {average_proposer_rewards} ETH')
    print(f'Average Base Fee: {average_total_base_fee_eth} ETH')

    #Subblock accumulation algorithm
    def proposer_rewards_with_accumulation(s, average_proposer_rewards, average_total_base_fee_eth):
        total_reward = 0
        accum_reward = 0  # Accumulated reward
        
        pred_rev = average_proposer_rewards + average_total_base_fee_eth

        for sub_slot in range(s + 1):

            # Add the split volume to the accumulated volume
            accum_reward += pred_rev / (s + 1)

            # Calculate potential reward and gas cost
            potential_reward = accum_reward 
            gas_cost = average_total_base_fee_eth

            # Check if a sandwich is profitable
            if potential_reward > gas_cost:
                total_reward += potential_reward - gas_cost
                accum_reward = 0  # Reset accumulated volume

        return total_reward

    # Calculate the proposer rewards for each sub-block setting (ignore s = 0)
    range_given = range(0, 40)  # Start from 1 to ignore x = 0
    proposer_rewards_list = [proposer_rewards_with_accumulation(s, average_proposer_rewards, average_total_base_fee_eth) for s in range_given]

    # Plotting splits against rewards
    plt.plot(range_given, proposer_rewards_list, '-o', label='Proposer Rewards')

    # Exponential decay fitting function
    def exponential_decay(x, a, b):
        return a * np.exp(-b * x) 

    # Fit the exponential decay model to the data
    params, _ = curve_fit(exponential_decay, range_given, proposer_rewards_list, p0=(1, 0.1))
    a, b = params

    # Print the equation
    print(f'Fit Equation: Complex proposer_rewards = {a:.4f} * exp(-{b:.4f} * s)')

    # Plot the exponential decay model
    y_pred = exponential_decay(np.array(range_given), a, b)
    plt.plot(range_given, y_pred, label='Exponential Decay Fit', color='grey')
    plt.xlabel('Number of Splits')
    plt.ylabel('Proposer Rewards (ETH)')
    plt.title(f'Complex Proposer Rewards vs. Number of Splits')
    plt.legend()
    plt.show()

def main():

    # To retrieve data, runn all processing functions
    #sandwich_processor()  # Added processing function for Sandwich
    #jit_processor()       # Added processing function for JIT Sandwich
    #atomic_processor()    # Added processing function for Atomic
    #searcher_processing() # Added processing function for Searcher

    # Execute the functions
    df_analysis_avg = classification_breakdown()
    sandwich_subblock_algo(df_analysis_avg)
    jit_subblock_algo(df_analysis_avg)
    atomic_subblock_algo(df_analysis_avg)
    cex_dex_algo(df_analysis_avg)
    searcher_subblock_algo(df_analysis_avg)

if __name__ == "__main__":
    main()  # Run the main function




























