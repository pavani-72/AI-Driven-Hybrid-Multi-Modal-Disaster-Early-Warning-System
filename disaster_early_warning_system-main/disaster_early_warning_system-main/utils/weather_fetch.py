import requests

API_KEY = "c2653e60394e02dbb312652f570b0a6d"


def get_weather_data(city):

    url = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={API_KEY}&units=metric"

    response = requests.get(url)
    data = response.json()

    weather_sequence = []

    for item in data["list"][:6]:

        temperature = item["main"]["temp"]

        rainfall = 0
        if "rain" in item and "3h" in item["rain"]:
            rainfall = item["rain"]["3h"]

        wind_speed = item["wind"]["speed"]

        weather_sequence.append([temperature, rainfall, wind_speed])

    return weather_sequence