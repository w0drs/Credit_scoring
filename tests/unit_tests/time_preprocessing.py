from datetime import datetime

from src.tools.time import datetime_to_bank_format

def test_time_preprocessing():
    time_now = datetime_to_bank_format()
    assert time_now

def test_list_of_times():
    list_of_times = ["2024-06-18", "2017-01-12", "2016-12-11", "2010-01-21"]
    for time in list_of_times:
        date_obj = datetime.strptime(time, "%Y-%m-%d")
        processed_time = datetime_to_bank_format(date_obj)
        assert processed_time
        print()
        print(f'{processed_time}')