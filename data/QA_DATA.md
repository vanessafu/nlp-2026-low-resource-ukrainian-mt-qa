# ZNO Scraped QA Data

Source: [zno.osvita.ua](https://zno.osvita.ua) — Ukraine's largest national exam (ЗНО / НМТ) preparation platform. Questions are organised by subject and topic tag, each with a structured correct answer and a detailed explanation written by subject experts.

---

## Files at a Glance

| File | Items | Tags | Topic sections | Size |
|------|------:|-----:|:--------------:|-----:|
| `data/raw/language.json` | 1,946 | 51 | 6 | 4.6 MB |
| `data/raw/language_literature.json` | 3,292 | 132 | 14 | 9.9 MB |
| `data/raw/history.json` | 2,553 | 47 | 8 | 5.7 MB |
| `data/raw/literature.json` | 1,153 | 40 | 7 | 2.4 MB |
| **Total** | **8,944** | **270** | | **22.6 MB** |

> `language_literature.json` covers both language and literature topics from a combined exam index page whose tag ID space is independent of `language.json` / `literature.json`. Content overlaps — do not merge by `tag_id`.

---

## Record Structure

Every item is a JSON object with the following fields:

```json
{
  "question":        "Question stem (Markdown)",
  "answers": [
    {"marker": "А", "text": "Option text"},
    {"marker": "Б", "text": "Option text"},
    {"marker": "В", "text": "Option text"},
    {"marker": "Г", "text": "Option text"}
  ],
  "correct_answers": ["А"],
  "explanation":     "Expert explanation (Markdown)",
  "subject":         "ukrainian-language",
  "tag_id":          796,
  "tag_name":        "Склад. Наголос",
  "topic":           "1.1. Фонетика. Графіка. Орфоепія (114 завдань)"
}
```

| Field | Notes |
|-------|-------|
| `question` | Stems often contain `**bold**` to mark highlighted letters (phonetics, orthography) |
| `answers` | Markers are Cyrillic: А / Б / В / Г / Д. 4-option and 5-option questions both exist |
| `correct_answers` | Always a single-element list; all retained items are single-choice |
| `explanation` | Full Markdown (paragraphs, bold, italic, lists). Empty string for ~4–10 % of items |
| `subject` | Set at scrape time; see per-file notes for known inconsistencies |
| `tag_id` | Website-internal topic ID; unique within a scrape run, not globally across files |
| `tag_name` | Short topic label, e.g. `"Склад. Наголос"` |
| `topic` | Parent section heading with the site's own question count in parentheses |

---

## Scraping Pipeline

### Tools

| Script | Role |
|--------|------|
| `scripts/scrape_zno_tags.py` | Crawls the index page, enumerates sub-tags, fetches and caches HTML, calls the parser |
| `scripts/convert_zno_html.py` | Parses a single HTML page into structured QA records; also usable as a standalone CLI |

### Steps

```
/tema.html  (subject index)
  └── <li class="tag-item main">               ← section heading (e.g. "1.1. Фонетика")
        └── <li class="tag-item"
               data-tag_id="N"
               data-cnt="N">                   ← sub-topic tag
               └── <a href="/subject/tag-slug/">
                     └── GET full page HTML
                           └── cache to data/raw/zno_html/<subject>/<slug>.html
                                 └── parse every <div class="task-card">
                                       └── append to data/raw/<subject>.json
```

**Caching:** each HTML page is downloaded once. Subsequent runs skip any `tag_id` already present in the output JSON, making runs resumable after interruption.

**Rate limiting:** 1.5 s delay between HTTP requests (configurable via `--delay`).

### Per-card HTML layout

```html
<div class="task-card card_{id}" id="q{n}">
  <form class="q-test">
    <input name="q[id]"  value="{question_id}">
    <input name="q[tip]" value="{type}">        <!-- question type -->
    <div class="question"> … </div>
    <div class="answers">
      <div class="answer">
        <span class="marker">А</span>  … option text …
      </div>
      …
    </div>
    <input name="result" value="{a|b|c|d|e}">   <!-- correct answer key -->
  </form>
  <div id="commentar_{question_id}"> … </div>    <!-- explanation -->
</div>
```

### Question-type filtering

The site encodes question type in the hidden `q[tip]` field:

| `tip` | Type | Action |
|-------|------|--------|
| `1` | Single-choice (А/Б/В/Г/Д) | **Kept** — fully parsed |
| `2` | Matching / correspondence | Parsed as `question_type: "matching"` with `correct_answers: ["А-1","Б-3",…]` |
| `4` | Open-ended text input | **Skipped** — no structured answer to extract |
| `5` | Ordering / sequencing | **Skipped** — variable layout, no canonical answer field |
| `7` | Other special formats | **Skipped** |

### HTML → Markdown conversion

**Question stem and answer options** (inline mode):
- `<b>`, `<i>`, `<u>`, `<strong>`, `<em>` → `**bold**`
  (on the site these mark phonetically or grammatically highlighted letters)
- All other tags: content extracted, tags dropped
- `<br>` → single space

**Explanation text** (block mode):
- `<strong>` / `<b>` → `**text**`
- `<em>` / `<i>` → `_text_`
- `<p>` → paragraph followed by a blank line
- `<br>` → newline
- `<ul>` / `<ol>` + `<li>` → Markdown list items
- Leading `<strong>Пояснення</strong>` header stripped automatically

---

## Per-file Details

### `data/raw/language.json` — Ukrainian Language (Grammar & Spelling)

**Source index:** `https://zno.osvita.ua/ukrmova/tema.html`

| Metric | Value |
|--------|-------|
| Total items | **1,946** |
| Sub-topic tags | 51 (`tag_id` 796–850) |
| With explanation | 1,943 (99.8 %) |
| 4-option questions | 756 |
| 5-option questions | 1,190 |
| `subject` field | `"ukrainian-language"` (see note below) |

**Items kept vs. skipped:**

| Result | Count |
|--------|------:|
| tip1 parsed successfully | 1,773 |
| tip4 skipped (open-ended) | 465 |
| tip7 skipped | 7 |
| tip1 structurally incomplete | 4 |

**Topic sections:**

| Section | Items |
|---------|------:|
| 1.1. Фонетика. Графіка. Орфоепія | 113 |
| 1.2. Орфографія | 360 |
| 1.3. Лексикологія | 200 |
| 1.4. Будова слова. Словотвір | 57 |
| 1.5. Морфологія | 551 |
| 1.6. Синтаксис. Пунктуація | 605 |
| *(no topic — legacy items)* | 60 |

> **Legacy items (60):** Parsed from manually downloaded HTML files in `data/raw/language/`
> before the scraper existed. These items lack `tag_id`, `tag_name`, and `topic` fields and
> carry the old subject label `"ukrainian-language-and-literature"`.
> An additional 113 items from a test scrape run carry `subject: "ukrmova"`.

---

### `data/raw/language_literature.json` — Ukrainian Language + Literature (Combined)

**Source index:** `https://zno.osvita.ua/ukrainian/tema.html`

| Metric | Value |
|--------|-------|
| Total items | **3,292** |
| Sub-topic tags | 132 |
| With explanation | 3,170 (96.3 %) |
| 4-option questions | 1,120 |
| 5-option questions | 2,172 |
| `subject` field | `"ukrainian-language-and-literature"` (all items) |

**Topic sections:**

| Section | Items |
|---------|------:|
| **Language** | |
| 1.1. Фонетика. Графіка. Орфоепія | 113 |
| 1.2. Орфографія | 355 |
| 1.3. Лексикологія | 200 |
| 1.4. Будова слова. Словотвір | 57 |
| 1.5. Морфологія | 551 |
| 1.6. Синтаксис. Пунктуація | 581 |
| 1.7. Стилістика. Текст. Розвиток мовлення | 297 |
| **Literature** | |
| 2.1. Усна народна творчість | 54 |
| 2.2. Давня українська literatura | 80 |
| 2.3. Лit. кінця XVIII – поч. XX ст. | 278 |
| 2.4. Лit. XX ст. | 478 |
| 2.5. Твори письменників-емігрантів | 52 |
| 2.6. Сучасний літературний процес | 40 |
| 2.7. Теорія літератури | 156 |

> This file covers both language and literature under one combined exam index. Its `tag_id`
> values are from a separate numbering space and **cannot be used to deduplicate** against
> `language.json` or `literature.json`.

---

### `data/raw/history.json` — Ukrainian History

**Source index:** `https://zno.osvita.ua/ukraine-history/tema.html`

| Metric | Value |
|--------|-------|
| Total items | **2,553** |
| Sub-topic tags | 47 |
| With explanation | 2,436 (95.4 %) |
| 4-option questions | 2,553 (100 %) |
| `subject` field | `"ukrainian-history"` |

**Items kept vs. skipped:**

| Result | Count |
|--------|------:|
| tip1 parsed successfully | 2,553 |
| tip4 skipped (open-ended) | 382 |
| tip5 skipped (ordering) | 271 |
| tip7 skipped | 212 |
| tip1 structurally incomplete | 27 |
| **Total on site** | **~3,445** |
| **Skip rate** | **~26 %** |

History has the highest skip count in absolute terms; tip5 (event ordering) and tip7
(map/image matching) are common in this subject.

**Topic sections:**

| Section | Items |
|---------|------:|
| 1.1. Найдавніші часи – перша пол. XVI ст. | 378 |
| 1.2. Друга пол. XVI ст. – перша пол. XVIII ст. | 428 |
| 1.3. Кінець XVIII ст. – XIX ст. | 453 |
| 2.1. 1914–1945 р. | 665 |
| 2.2. 1945 р. – поч. XXI ст. | 450 |
| 3.1. Персоналії | 79 |
| 3.2. Архітектура | 49 |
| 3.3. Мистецтво | 51 |

---

### `data/raw/literature.json` — Ukrainian Literature

**Source index:** `https://zno.osvita.ua/ukrlit/tema.html`

| Metric | Value |
|--------|-------|
| Total items | **1,153** |
| Sub-topic tags | 40 |
| With explanation | 1,037 (90.0 %) |
| 4-option questions | 96 |
| 5-option questions | 1,057 |
| `subject` field | `"ukrainian-literature"` |

**Items kept vs. skipped:**

| Result | Count |
|--------|------:|
| tip1 parsed successfully | 1,153 |
| tip4 skipped (open-ended) | 785 |
| tip1 structurally incomplete | 1 |
| **Total on site** | **~1,939** |
| **Skip rate** | **~41 %** |

Literature has the highest skip rate: over 40 % of on-site questions are open-ended (tip4),
typically asking students to quote or complete a passage from a literary text.

**Topic sections:**

| Section | Items |
|---------|------:|
| 1.1. Усна народна творчість | 54 |
| 1.2. Давня українська literatura | 80 |
| 1.3. Лit. кінця XVIII – поч. XX ст. | 293 |
| 1.4. Лit. XX ст. | 478 |
| 1.5. Твори письменників-емігрантів | 52 |
| 1.6. Сучасний літературний процес | 40 |
| 1.7. Теорія літератури | 156 |

---

## Known Issues

1. **Inconsistent `subject` values in `language.json`.**
   The file contains items from three separate collection runs:

   | Items | `subject` value | Origin |
   |------:|-----------------|--------|
   | 60 | `"ukrainian-language-and-literature"` | Manual HTML download (pre-scraper) |
   | 113 | `"ukrmova"` | First test scrape run |
   | 1,773 | `"ukrainian-language"` | Full scrape run |

2. **Legacy items in `language.json` missing metadata.**
   The 60 manually collected items have no `tag_id`, `tag_name`, or `topic` fields.

3. **Content overlap between `language_literature.json` and the individual files.**
   The combined index and the per-subject indexes serve different versions of the exam.
   Both cover the same curriculum but with different tag IDs and partially different question sets.
   Use question-stem text hashing to deduplicate if merging.

4. **Skipped question HTML is preserved.**
   All downloaded pages are cached under `data/raw/zno_html/<subject>/`. If tip4/5 parsing
   is implemented later, no re-download is required.

5. **Explanations reference an answer that may already be marked.**
   Explanations are written from the perspective of justifying the correct answer only;
   wrong options are not individually explained.
