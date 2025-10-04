---
title: MedVietAI Processing
emoji: ⚕️
colorFrom: green
colorTo: pink
sdk: docker
pinned: false
license: apache-2.0
short_description: Data processing with en-vi translation. Derived from 500k mi
---

## 🚀 Quick Access

[HF Space](https://huggingface.co/spaces/MedVietAI/processing)

[MedDialog-100k](https://huggingface.co/datasets/MedAI-COS30018/MedDialog-EN-100k)

[MedDialog-10k](https://huggingface.co/datasets/MedAI-COS30018/MedDialog-EN-10k)

[PubMedQA-Labelled](https://huggingface.co/datasets/MedAI-COS30018/PubMedQA-L)

[PubMedQA-Unlabelled](https://huggingface.co/datasets/MedAI-COS30018/PubMedQA-U)

[PubMedQA-Mapper](https://huggingface.co/datasets/MedAI-COS30018/PubMedQA-MAP)

## 🎯 Features

### 🔄 Advanced Data Augmentation
- **Paraphrasing**: Multi-model rotation (NVIDIA + Gemini) with easy/hard difficulty levels
- **Backtranslation**: Vietnamese pivot language for semantic preservation
- **Style Standardization**: Clinical voice enforcement and professional medical tone
- **Response Validation**: Invalid response detection and retry logic (max 3 attempts)
- **Quality Guards**: Length/semantic validation for backtranslation outputs

### 🇻🇳 Vietnamese Translation
- **Complete Translation**: All text fields translated when Vietnamese mode is enabled
- **Quality Validation**: Translation quality checks with fallback to original text
- **SFT Format**: `instruction`, `input`, `output` fields translated
- **RAG Format**: `question`, `answer`, `context` fields translated
- **Sanitization**: Repetition reduction and whitespace normalization

### 📊 SFT Data Enrichment
- **Multiple Answer Variants**: 2-3 different answers per question for better reasoning
- **Multiple Question Variants**: 2-3 different questions per answer for diverse training
- **Cross Combinations**: All question × answer variant combinations (up to 9 per sample)
- **Vietnamese Variants**: Translated versions of enriched combinations
- **Reasoning Enhancement**: Multiple reasoning paths for improved model training

### 🔍 Quality Assurance
- **Invalid Response Detection**: Catches "Fail", "Invalid", "I can't", "Sorry", etc.
- **Retry Logic**: Up to 3 attempts with different paraphrasing difficulties
- **Drop Strategy**: Samples dropped if retry fails (no fallback answers)
- **Consistency Checking**: LLM-based validation of answer quality
- **De-identification**: PHI removal with configurable strictness

### 🎯 RAG Optimization
- **Embedding-Friendly**: Concise, direct text optimized for dense retrieval
- **Context Generation**: Synthetic context creation when missing
- **Content Cleaning**: Conversational element removal for medical focus
- **Length Control**: Hard caps on question/answer/context lengths
- **Quality Filtering**: Invalid response cleaning for RAG corpora

## 📋 Supported Datasets

### Medical Dialogue
- **HealthCareMagic**: 100k medical conversations
- **iCliniq**: 10k derived medical Q&A

### Biomedical QA
- **PubMedQA-L**: Labeled biomedical questions
- **PubMedQA-U**: Unlabeled biomedical questions  
- **PubMedQA-MAP**: Mapped biomedical Q&A pairs

## ⚙️ Configuration

### Augmentation Parameters
```python
class AugmentOptions:
    paraphrase_ratio: float = 0.2          # 0.0-1.0
    paraphrase_outputs: bool = True         # Augment model answers
    backtranslate_ratio: float = 0.1        # 0.0-1.0 (Vietnamese pivot)
    style_standardize: bool = True          # Enforce clinical style
    deidentify: bool = True                 # Remove PHI
    dedupe: bool = True                     # Remove duplicates
    max_chars: int = 5000                   # Text length limit
    consistency_check_ratio: float = 0.05   # 0.0-1.0
    expand: bool = True                     # Enable enrichment
    max_aug_per_sample: int = 2             # 1-3 variants
```

### Processing Modes
- **SFT Processing**: Supervised Fine-Tuning format with enrichment
- **RAG Processing**: Question-Context-Answer format for retrieval
- **Vietnamese Mode**: Complete translation of all text fields

## 📈 Output Statistics

The system tracks comprehensive statistics:
- `written`: Successfully processed samples
- `paraphrased_input/output`: Paraphrasing counts
- `backtranslated_input/output`: Backtranslation counts
- `dropped_invalid`: Samples dropped due to failed retries
- `vietnamese_variants`: Vietnamese variants created
- `dedup_skipped`: Duplicate samples removed
- `consistency_failed`: Samples flagged for quality issues

## 🔧 Usage

### Web Interface
1. Visit the [HF Space](https://huggingface.co/spaces/MedVietAI/processing)
2. Select dataset and processing mode (SFT/RAG)
3. Enable Vietnamese translation if needed
4. Click process button

### API Usage
```bash
# SFT Processing with Vietnamese translation
curl -X POST "https://huggingface.co/spaces/MedVietAI/processing/process/healthcaremagic" \
  -H "Content-Type: application/json" \
  -d '{
    "augment": {
      "paraphrase_ratio": 0.2,
      "backtranslate_ratio": 0.1,
      "paraphrase_outputs": true,
      "style_standardize": true,
      "deidentify": true,
      "dedupe": true,
      "expand": true
    },
    "vietnamese_translation": true
  }'

# RAG Processing
curl -X POST "https://huggingface.co/spaces/MedVietAI/processing/rag/healthcaremagic" \
  -H "Content-Type: application/json" \
  -d '{
    "vietnamese_translation": true
  }'
```

## 📚 Documentation

- [Request Documentation](https://huggingface.co/spaces/MedVietAI/processing/blob/main/REQUEST.md)
- [Data Processing Guide](https://huggingface.co/spaces/MedVietAI/processing/blob/main/DATA_PROCESSING.md)

## 📄 License

[Apache-2.0 LICENSE](https://huggingface.co/spaces/MedVietAI/processing/blob/main/LICENSE.txt)

