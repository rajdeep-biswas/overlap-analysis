import requests
from custom_stopword_lists import custom_stopwords


def get_timezones(timezones_json_url = 'https://raw.githubusercontent.com/dmfilipenko/timezones.json/master/timezones.json'):

    return [item['abbr'].lower() for item in requests.get(url=timezones_json_url).json()]


def populate_custom_stopwords():

    custom_stopwords.extend(get_timezones())
    return custom_stopwords