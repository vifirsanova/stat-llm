#!/usr/bin/env python3
import pandas as pd
import re


def syllables(text):
    vowel = 'аеёиоуыэюя'
    cons = 'бвгджзлмнрсхцчшщ'
    brief = 'кпстф'
    voiced = 'й'
    deaf = 'ьъ'
    other = 'бвгджзйклмнпрсстфхцчшщ'

    def _is_not_last_sep(txt):
        return any(ch in vowel for ch in txt)

    def _add_sep(force_sep):
        nonlocal current_syllable, syllables
        if force_sep:
            current_syllable += ' '
            return

        if not re.search(f'[{vowel}]', current_syllable):
            return

        syllables.append(current_syllable)
        current_syllable = ''

    syllables = []
    current_syllable = ''
    words = text.split()

    for word in words:
        if len(word) < 2:
            word = word.replace('-', '')

        for i, char in enumerate(word):
            current_syllable += char
            next_char = word[i+1] if i+1 < len(word) else ''

            if not next_char or next_char not in 'абвгдеёжзийклмнопрстуфхцчшщыьэюя':
                continue

            if (i != 0 and i != len(word)-1 and char in voiced and
                _is_not_last_sep(word[i+1:])):
                _add_sep(False)
                continue

            if (i < len(word)-1 and char in vowel and
                next_char in vowel):
                _add_sep(False)
                continue

            if (i < len(word)-2 and char in vowel and
                word[i+1] in other and word[i+2] in vowel):
                _add_sep(False)
                continue

            if (i < len(word)-2 and char in vowel and
                word[i+1] in brief and word[i+2] in other and
                _is_not_last_sep(word[i+1:])):
                _add_sep(False)
                continue

            if (i > 0 and i < len(word)-1 and char in cons and
                word[i-1] in vowel and next_char not in vowel and
                next_char not in deaf and
                _is_not_last_sep(word[i+1:])):
                _add_sep(False)
                continue

            if (i < len(word)-1 and char in deaf and
                (next_char not in vowel or _is_not_last_sep(word[:i]))):
                _add_sep(False)
                continue

        _add_sep(True)

    if current_syllable:
        syllables.append(current_syllable)

    return '-'.join(syllables)

with open("morphs_output.txt", "r", encoding="utf-8") as morphs:
    words = morphs.read().strip().split('\n')

word_series = pd.Series(words, name = 'words')
df = pd.DataFrame({
    'words': word_series,
    'hyphenated_words': word_series
})

df['hyphenated_words'] = df['hyphenated_words'].apply(lambda x: syllables(x))

df.to_csv('/content/syllables.csv', index=False, encoding='utf-8')

