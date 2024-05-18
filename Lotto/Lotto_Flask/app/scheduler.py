import requests

def fetch_lottery_results():
    api_url = 'https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo=1119'

    try:
        response = requests.get(api_url)
        data = response.json()
        
        winning_numbers = {
            'drawNo': data['drwNo'],
            'numbers': [data[f'drwtNo{i}'] for i in range(1, 7)],
            'bonusNumber': data['bnusNo']
        }

        print(f"Fetched lottery results: {winning_numbers}")

    except Exception as e:
        print(f"Error fetching lottery results: {e}")
