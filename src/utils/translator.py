import json
import os

class Translator:
    def __init__(self, lang='en'):
        self.lang = lang
        self.translations = self._load_translations()

    def _load_translations(self):
        path = os.path.join('locales', f'{self.lang}.json')
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get(self, key, **kwargs):
        text = self.translations.get(key, key)
        return text.format(**kwargs)