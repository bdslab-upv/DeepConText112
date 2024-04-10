"""
DESCRIPTION: functions to prepare data.
AUTHOR: Pablo Ferri-Borredà
DATE: 01/07/22
"""

# MODULES IMPORT
import re

from pandas import DataFrame, Series
from transformers import DistilBertTokenizer

# SETTINGS
TEXT_COLUMN = 'TEXT'
TOKENIZER_TYPE = 'DistilBERT'


# DATA PREPARATION
# Main function for data preparation
def prepare_data(data: DataFrame) -> DataFrame:
    # Date variables generation
    data = _generate_date_variables(data)

    # Text preparation
    data = _prepare_text(data)

    # Columns selection and sorting
    # columns definition
    columns2keep = ['YEAR_MONTH', 'YEAR', TEXT_COLUMN, TEXT_COLUMN + '_PROC', TEXT_COLUMN + '_TOK',
                    TEXT_COLUMN + '_IDX', TEXT_COLUMN + '_ATM', 'LIFE_THREATENING_NO', 'LIFE_THREATENING_YES']
    # selection and sorting
    data = data[columns2keep]

    # Output
    return data


# Date variables generation
def _generate_date_variables(data: DataFrame) -> DataFrame:
    # Year
    year = data['DATE_TIME'].apply(lambda single_date: single_date.year)

    # Month
    month = data['DATE_TIME'].apply(lambda single_date: single_date.month)

    # Year
    data['YEAR'] = year

    # Year-month
    data['YEAR_MONTH'] = _generate_year_month_variable(year, month)

    # Output
    return data


# Year-month variable generation
def _generate_year_month_variable(year_series: Series, month_series: Series) -> Series:
    # Casting and formatting
    # year
    year_series_str = year_series.astype(str)
    # month
    month_series_str = month_series.astype(str)
    month_series_str = month_series_str.apply(lambda x: '0' + x if len(x) == 1 else x)

    # Variable generation
    year_month_series = year_series_str + month_series_str
    year_month_series = year_month_series.astype(int)

    # Output
    return year_month_series


# Text preparation
def _prepare_text(data: DataFrame) -> DataFrame:
    # Pretokenization processing
    data[TEXT_COLUMN + '_PROC'] = data[TEXT_COLUMN].apply(_pretokenization_processing)

    # Text formatting
    text_list = data[TEXT_COLUMN + '_PROC'].to_list()

    # Tokenizer initialization
    if TOKENIZER_TYPE == 'DistilBERT':
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
    else:
        raise ValueError('Unrecognized model type identifier.')

    # Text tokenization
    text_tokenized = data[TEXT_COLUMN + '_PROC'].apply(tokenizer.tokenize)

    # Text encoding
    text_encoded = tokenizer(text_list, padding=True, truncation=True)

    # Arrangement
    data[TEXT_COLUMN + '_TOK'] = text_tokenized
    data[TEXT_COLUMN + '_IDX'] = text_encoded['input_ids']
    data[TEXT_COLUMN + '_ATM'] = text_encoded['attention_mask']

    # Output
    return data


# Pretokenization processing
# settings
accent_marks_map = {'á': 'a', 'à': 'a', 'é': 'e', 'è': 'e', 'í': 'i', 'ì': 'i', 'ó': 'o', 'ò': 'o', 'ú': 'u', 'ù': 'u'}
special_characters_map = {'-': ' ', '.': ' ', ',': ' ', '/': ' ', '\\': ' '}


# processing
def _pretokenization_processing(text_str: str) -> str:
    # Lower casing
    text_str_lower = text_str.lower()

    # Accent marks filtering
    # initialization
    text_str_proc = ''
    # filtering
    for char in text_str_lower:
        if char not in accent_marks_map.keys():
            text_str_proc += char
        else:
            text_str_proc += accent_marks_map[char]

    # Numbers and text splitting
    if _has_numbers(text_str_proc):
        text_list = re.split('(\d+)', text_str_proc)
        text_str_proc = ' '.join(text_list)

    # Special characters dropping
    # initialization
    text_str_processed = ''
    # filtering
    for char in text_str_proc:
        if char not in special_characters_map.keys():
            text_str_processed += char
        else:
            text_str_processed += special_characters_map[char]

    # Multiple space removal
    text_str_processed = re.sub(' +', ' ', text_str_processed)

    # Output
    return text_str_processed


# Digit detection
def _has_numbers(token: str) -> bool:
    return any(char.isdigit() for char in token)
