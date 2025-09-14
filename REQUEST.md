# 📑 MedAI Processing – Request Examples

Base URL of the Space:  
**`https://binkhoale1812-medai-processing.hf.space`**

This Space processes medical datasets into a centralised fine-tuning format (JSONL + CSV) with optional augmentations such as **paraphrasing**, **back-translation**, **style standardisation**, **de-identification**, and **deduplication**.  

---

## 🔹 1. Process HealthCareMagic

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
        "augment": {
          "paraphrase_ratio": 0.1,
          "backtranslate_ratio": 0.05,
          "paraphrase_outputs": false,
          "style_standardize": true,
          "deidentify": true,
          "dedupe": true,
          "max_chars": 5000
        },
        "sample_limit": 2000,
        "seed": 42
      }' \
  https://binkhoale1812-medai-processing.hf.space/process/healthcaremagic
````

---

## 🔹 2. Process iCliniq

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
        "augment": {
          "paraphrase_ratio": 0.2,
          "backtranslate_ratio": 0.1,
          "paraphrase_outputs": true,
          "style_standardize": true,
          "deidentify": true,
          "dedupe": true,
          "max_chars": 5000
        },
        "sample_limit": 1500,
        "seed": 123
      }' \
  https://binkhoale1812-medai-processing.hf.space/process/icliniq
```

---

## 🔹 3. Process PubMedQA (Labelled)

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
        "augment": {
          "paraphrase_ratio": 0.05,
          "backtranslate_ratio": 0.02,
          "paraphrase_outputs": false,
          "style_standardize": true,
          "deidentify": false,
          "dedupe": true,
          "max_chars": 8000
        },
        "sample_limit": 1000,
        "seed": 99
      }' \
  https://binkhoale1812-medai-processing.hf.space/process/pubmedqa_l
```

---

## 🔹 4. Process PubMedQA (Unlabelled)

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
        "augment": {
          "paraphrase_ratio": 0.05,
          "backtranslate_ratio": 0.05,
          "paraphrase_outputs": false,
          "style_standardize": true,
          "deidentify": true,
          "dedupe": true,
          "max_chars": 7000,
          "consistency_check_ratio": 0.01,
          "distill_fraction": 0.1
        },
        "sample_limit": 500,
        "seed": 7
      }' \
  https://binkhoale1812-medai-processing.hf.space/process/pubmedqa_u
```

---

## 🔹 5. Process PubMedQA (Map)

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
        "augment": {
          "paraphrase_ratio": 0.1,
          "backtranslate_ratio": 0.05,
          "paraphrase_outputs": true,
          "style_standardize": true,
          "deidentify": true,
          "dedupe": true,
          "max_chars": 6000
        },
        "sample_limit": 1200,
        "seed": 2024
      }' \
  https://binkhoale1812-medai-processing.hf.space/process/pubmedqa_map
```

---

## 🔹 6. Check Current Job Status

```bash
curl https://binkhoale1812-medai-processing.hf.space/status
```

---

## 🔹 7. List Generated Artifacts

```bash
curl https://binkhoale1812-medai-processing.hf.space/files
```

---

# ✅ Notes

* Each run outputs both `.jsonl` and `.csv` in `cache/outputs/` and also uploads them to Google Drive folder ID:
  `1JvW7its63E58fLxurH8ZdhxzdpcMrMbt`
* `augment` options can be adjusted per dataset:

  * `paraphrase_ratio` – % of rows paraphrased (0–1)
  * `backtranslate_ratio` – % of rows back-translated
  * `paraphrase_outputs` – whether to also augment model answers
  * `style_standardize` – enforce neutral, clinical style
  * `deidentify` – redact PHI (emails, phones, URLs, IPs)
  * `dedupe` – skip duplicate pairs
  * `consistency_check_ratio` – run lightweight QA sanity check
  * `distill_fraction` – generate pseudo-labels for unlabelled data
