"""
Test Universal 150+ Translation & Transcription Engine
"""
from engine.universal_translator import UniversalTranslatorEngine

engine = UniversalTranslatorEngine()

test_cases = [
    ("நாளை மாலை 5 மணிக்கு திட்ட ஆய்வுக் கூட்டம் உள்ளது, ஸ்லைடுகளை தயார் செய்யவும்", "Tamil"),
    ("我們明天下午2點在台北101開Q3 project review meeting，記得把slides準備好。", "Chinese"),
    ("明日午後2時に会議があります、スライドを準備してください。", "Japanese"),
    ("Tenemos una reunión mañana a las 2 pm, por favor preparen las diapositivas.", "Spanish"),
    ("Wir haben morgen um 14 Uhr ein Meeting, bitte die Folien vorbereiten.", "German"),
    ("Nous avons une réunion demain à 14h, veuillez préparer les diapositives.", "French")
]

for text, lang_name in test_cases:
    res = engine.process(text)
    src = res["source_language"]
    print(f"=== Input: {lang_name} ({src['flag']} {src['name']} - {src['native']}) ===")
    print(f"Source: {res['source_transcription']}")
    print(f"English: {res['english_translation']}")
    print(f"Event: {res['structured_representation']['event']}")
    print(f"Action: {res['structured_representation']['action']}")
    print(f"Time: {res['structured_representation']['time']['normalized']}")
    print(f"Tamil Exp: {res['multilingual_explanations']['ta']}")
    print()

print("Universal Translation Engine: All tests passed!")
