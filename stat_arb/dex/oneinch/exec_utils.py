import os
import asyncio
import requests
import json
from web3 import Web3
from dotenv import load_dotenv

# Load environment variables
load_dotenv() 
chainId = 42161 
web3RpcUrl = os.getenv("RPC_URL") 
api_key = os.getenv("API_KEY")
walletAddress = os.getenv("WALLET_ADDRESS")
privateKey = os.getenv("PRIVATE_KEY")
headers = { "Authorization": f"Bearer {api_key}", "accept": "application/json" }
apiBaseUrl = f"https://api.1inch.dev/swap/v6.0/{chainId}"
broadcast_api_url = f"https://api.1inch.dev/tx-gateway/v1.1/{chainId}/broadcast"
web3 = Web3(Web3.HTTPProvider(web3RpcUrl))
walletAddress = web3.to_checksum_address(walletAddress)

def apiRequestUrl(methodName, queryParams):
    return f"{apiBaseUrl}{methodName}?{'&'.join([f'{key}={value}' for key, value in queryParams.items()])}"

def checkAllowance(tokenAddress, walletAddress):
    """Check token allowance"""
    url = apiRequestUrl("/approve/allowance", {"tokenAddress": tokenAddress, "walletAddress": walletAddress})
    response = requests.get(url, headers=headers)
    data = response.json()
    return data["allowance"]

async def broadCastRawTransaction(raw_transaction):
    """Broadcast a raw transaction"""
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": api_key,  
        }
        body = json.dumps({"rawTransaction": raw_transaction})

        response = requests.post(broadcast_api_url, data=body, headers=headers)
        print(f"Broadcast response: {response.status_code}, {response.text}")

        if response.status_code == 200:
            res = response.json()
            return res.get('transactionHash')
        else:
            print(f"Error broadcasting transaction: {response.status_code}, {response.text}")
            return None
    except Exception as e:
        print(f"Error broadcasting transaction: {e}")
        return None
    
async def signAndSendTransaction(transaction):
    """Sign and send a transaction"""
    nonce = web3.eth.get_transaction_count(walletAddress)
    transaction['nonce'] = nonce
    transaction['gasPrice'] = int(transaction['gasPrice'])
    transaction['value'] = int(transaction['value'])
    transaction['to'] = web3.to_checksum_address(transaction['to'])
    transaction['from'] = walletAddress

    signed_transaction = web3.eth.account.sign_transaction(transaction, privateKey)
    return await broadCastRawTransaction(signed_transaction.rawTransaction.hex())

async def estimateGasLimit(transaction):
    """Estimate gas limit for a transaction"""
    gas_price = transaction["gasPrice"]
    to_address = transaction["to"]
    value = transaction["value"]
    
    wallet_address = web3.to_checksum_address(walletAddress)
    to_address = web3.to_checksum_address(to_address)
    
    try:
        gas_limit = web3.eth.estimate_gas({
            "from": wallet_address,
            "to": to_address,
            "value": value,
            "data": transaction["data"],
            "gasPrice": gas_price
        })
    except Exception as e:
        print(f"Error estimating gas: {e}")
        gas_limit = 50000
    return gas_limit
    
async def buildTxForApproveTradeWithRouter(token_address, amount=None):
    """Build approval transaction"""
    url = apiRequestUrl("/approve/transaction", {"tokenAddress": token_address, "amount": amount} if amount else {"tokenAddress": token_address})
    response = requests.get(url, headers=headers)
    transaction = response.json()
    gas_limit = await estimateGasLimit(transaction)
    return {**transaction, "gas": gas_limit}

def buildTxForSwap(swapParams):
    """Build swap transaction"""
    url = apiRequestUrl("/swap", swapParams)
    print(f"Swap URL: {url}")
    print(f"Swap Parameters: {swapParams}")
    response = requests.get(url, headers={'Authorization': api_key})
    print(f"Response Status: {response.status_code}")
    print(f"Response Headers: {response.headers}")
    print(f"Response Text: {response.text}")
    swapTransaction = response.json()["tx"]
    return swapTransaction

async def execute_swap_transaction(swap_params):
    """Execute a complete swap transaction including approval and swap"""
    # Check allowance
    allowance = checkAllowance(swap_params["src"], walletAddress)
    print("Allowance: ", int(allowance))
    
    # If allowance is insufficient, approve
    if int(allowance) < int(swap_params["amount"]):
        print("Insufficient allowance, approving...")
        transactionForSign = await buildTxForApproveTradeWithRouter(swap_params["src"])
        print("Prepared Transaction for Approval:", transactionForSign)
        approveTxHash = await signAndSendTransaction(transactionForSign)
        print("Approve tx hash: ", approveTxHash)
    
    # Execute the swap
    swapTransaction = buildTxForSwap(swap_params)
    print("Transaction for swap: ", swapTransaction)
    swapTxHash = await signAndSendTransaction(swapTransaction)
    print("Swap tx hash: ", swapTxHash)
    
    return swapTxHash

# Run the main function
if __name__ == "__main__":
    asyncio.run(execute_swap_transaction(swapParams)) 