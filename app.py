import streamlit as st
import json
import re
import time
import textwrap

import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import google.generativeai as genai
from newspaper import Article

genai.configure(api_key="xxxxxx")
model = genai.GenerativeModel("gemini-2.0-flash")


@st.cache_resource
def load_ner_pipelines():
    pipes = {}

    pipes["English"] = spacy.load("en_core_web_lg")

    hi_tok = AutoTokenizer.from_pretrained("ai4bharat/IndicNER")
    hi_mod = AutoModelForTokenClassification.from_pretrained(
        "ai4bharat/IndicNER")
    pipes["Hindi"] = pipeline(
        "ner", model=hi_mod, tokenizer=hi_tok, aggregation_strategy="simple")

    ur_tok = AutoTokenizer.from_pretrained("mirfan899/urdu-bert-ner")
    ur_mod = AutoModelForTokenClassification.from_pretrained(
        "mirfan899/urdu-bert-ner")
    pipes["Urdu"] = pipeline(
        "ner", model=ur_mod, tokenizer=ur_tok, aggregation_strategy="simple")

    ar_tok = AutoTokenizer.from_pretrained(
        "CAMeL-Lab/bert-base-arabic-camelbert-mix-ner")
    ar_mod = AutoModelForTokenClassification.from_pretrained(
        "CAMeL-Lab/bert-base-arabic-camelbert-mix-ner")
    pipes["Arabic"] = pipeline(
        "ner", model=ar_mod, tokenizer=ar_tok, aggregation_strategy="simple")

    return pipes


def build_label_prompt(lang: str, text: str, entities: list[str]) -> str:
    entity_list = json.dumps(entities, ensure_ascii=False)
    if lang == "English":
        return f"""
You are a narrative understanding assistant.

Given the following text and list of entities, assign each entity one of:
 - Protagonist
 - Antagonist
 - Neutral

Reply ONLY with valid JSON (no explanation), e.g.:

[
  {{ "entity": "Entity Name", "label": "Protagonist" }},
  ...
]

Text: {text}

Entities: {entity_list}
"""
    if lang == "Hindi":
        return f"""
आप एक narrative understanding सहायक हैं।

निम्नलिखित समाचार पाठ और संस्थाओं की सूची के आधार पर प्रत्येक संस्था को एक भूमिका दें:
- Protagonist
- Antagonist
- Neutral

केवल मान्य JSON में उत्तर दें, कोई व्याख्या नहीं। उदाहरण:
[
  {{ "entity": "संस्था का नाम", "label": "Protagonist" }},
  ...
]

समाचार: {text}

संस्थाएं: {entity_list}
"""
    if lang == "Urdu":
        return f"""
آپ ایک بیانیہ فہمی اسسٹنٹ ہیں۔

مندرجہ ذیل خبر اور اداروں کی فہرست کی بنیاد پر ہر ادارے کو ایک کردار تفویض کریں:
- Protagonist
- Antagonist
- Neutral

صرف صحیح JSON میں جواب دیں، کوئی وضاحت نہیں:
[
  {{ "entity": "ادارے کا نام", "label": "Protagonist" }},
  ...
]

خبر: {text}

ادارے: {entity_list}
"""
    if lang == "Arabic":
        return f"""
أنت مساعد لفهم السرد.

بناءً على النص التالي وقائمة الكيانات، صنف كل كيان إلى:
- Protagonist
- Antagonist
- Neutral

أجب فقط بصيغة JSON صالحة، دون شرح. مثال:
[
  {{ "entity": "اسم الكيان", "label": "Protagonist" }},
  ...
]

النص: {text}

الكيانات: {entity_list}
"""
    raise ValueError(f"Unsupported language: {lang}")


def build_narrative_prompt(lang: str, text: str, entities: list[str]) -> str:
    entity_list = json.dumps(entities, ensure_ascii=False)
    if lang == "English":
        return f"""
You are a narrative extraction assistant.

From the text below, extract all complete events.  
Each event must have:
- subject (choose only from the entities list)
- action (verb)
- object (choose only from the entities list)
- context (optional extra details)

Reply ONLY with a JSON array of objects, e.g.:
[
  {{
    "subject": "...",
    "action": "...",
    "object": "...",
    "context": "..."
  }}
]

Text: {text}

Entities: {entity_list}
"""
    if lang == "Hindi":
        return f"""..."""
    if lang == "Urdu":
        return f"""..."""
    if lang == "Arabic":
        return f"""..."""
    raise ValueError(f"Unsupported language: {lang}")


@st.cache_data
def extract_narrative(text: str, lang: str):
    pipes = load_ner_pipelines()
    ner_pipe = pipes[lang]

    def chunk_text(t, width=450):
        return textwrap.wrap(t, width)

    raw_entities = []
    if lang == "English":
        doc = ner_pipe(text)
        raw_entities = [(ent.text, ent.label_) for ent in doc.ents]
    else:
        for chunk in chunk_text(text):
            for ent in ner_pipe(chunk):
                raw_entities.append((ent["word"], ent["entity_group"]))

    entities_dict: dict[str, list[str]] = {}
    for word, label in raw_entities:
        entities_dict.setdefault(label, []).append(word)
    all_entities = list({w for lst in entities_dict.values() for w in lst})

    label_prompt = build_label_prompt(lang, text, all_entities)
    r1 = model.generate_content([{"text": label_prompt}]).text.strip()
    r1 = re.sub(r"^```json\s*|```$", "", r1, flags=re.MULTILINE)
    try:
        labeled = json.loads(r1)
    except json.JSONDecodeError:
        labeled = []
    if not isinstance(labeled, list):
        labeled = []
    labeled_map = {
        e.get("entity", ""): e.get("label", "")
        for e in labeled
        if isinstance(e, dict) and "entity" in e and "label" in e
    }

    narrative_prompt = build_narrative_prompt(lang, text, all_entities)
    r2 = model.generate_content([{"text": narrative_prompt}]).text.strip()
    r2 = re.sub(r"^```json\s*|```$", "", r2, flags=re.MULTILINE)
    try:
        events = json.loads(r2)
    except json.JSONDecodeError:
        events = []
    if not isinstance(events, list):
        events = []

    def find_role(name: str) -> str:
        return labeled_map.get(name, "Unknown")

    for ev in events:
        ev["subject_role"] = find_role(ev.get("subject", ""))
        ev["object_role"] = find_role(ev.get("object", ""))

    return entities_dict, labeled, events


st.set_page_config(page_title="Narrative Extraction Demo", layout="wide")
st.title("🌐 Multilingual Narrative Extraction")

lang = st.selectbox("Choose language", ["English", "Hindi", "Urdu", "Arabic"])
input_mode = st.radio("Choose input method", ["Paste Text", "Scrape from URL"])

text = ""
if input_mode == "Paste Text":
    text = st.text_area("Paste your article text here:", height=250)
else:
    url = st.text_input("Enter URL to scrape:")
    if url:
        with st.spinner("Scraping article..."):
            try:
                article = Article(url)
                article.download()
                article.parse()
                text = article.text
                st.success("Article scraped successfully!")
                st.text_area("Scraped Article", value=text, height=250)
            except Exception as e:
                st.error(f"Failed to scrape article: {e}")

if st.button("Extract Narrative"):
    if not text.strip():
        st.warning("Please paste some article text or scrape a URL.")
    else:
        with st.spinner("Running NER → Role Labeling → Narrative Extraction..."):
            entities_dict, labeled, events = extract_narrative(text, lang)
            time.sleep(0.5)

        st.subheader("🕵️ Named Entities")
        st.json(entities_dict)

        st.subheader("🏷️ Labeled Entities")
        st.json(labeled)

        st.subheader("📖 Extracted Events")
        st.json(events)
