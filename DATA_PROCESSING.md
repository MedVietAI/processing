# 📊 MedAI Data Processing Techniques

This document comprehensively outlines all the data processing techniques implemented in the MedAI Processing project for augmenting and centrally processing medical datasets for LLM fine-tuning.

## 🎯 Project Overview

The MedAI Processing system transforms raw medical datasets into a **centralized fine-tuning format** (JSONL + CSV) with comprehensive data augmentation capabilities. The system processes multiple medical dataset types and applies various enhancement techniques to improve data quality and diversity.

## 🏗️ System Architecture

### Core Components
- **FastAPI Web Service**: RESTful API for dataset processing
- **Multi-LLM Rotator**: NVIDIA API + Google Gemini integration
- **Centralized Writer**: Parallel JSONL + CSV output generation
- **Google Drive Integration**: Automated artifact storage
- **Progress Monitoring**: Real-time job status tracking

### Supported Datasets
1. **HealthCareMagic** (100k medical dialogues)
2. **iCliniq** (10k medical consultations)
3. **PubMedQA-Labelled** (biomedical Q&A with answers)
4. **PubMedQA-Unlabelled** (biomedical Q&A without answers)
5. **PubMedQA-Map** (biomedical Q&A mapping format)

## 🔧 Data Processing Pipeline

### 1. Data Ingestion & Download
- **Hugging Face Hub Integration**: Automatic dataset downloading
- **Format Detection**: JSON/JSONL auto-detection and parsing
- **Caching System**: Local storage with symlink optimization

### 2. Data Cleaning & Preprocessing

#### Text Normalization
- **Unicode Fixing**: `ftfy` library for text encoding issues
- **Whitespace Standardization**: Consistent spacing and line breaks
- **Quote Canonicalization**: Standard quote character conversion
- **Terminal Punctuation**: Ensures proper sentence endings

#### Content Sanitization
- **Length Capping**: Configurable maximum character limits (default: 5000)
- **Language Detection**: English language validation using `langid`
- **Content Truncation**: Smart sentence boundary cutting for long texts

### 3. Data Augmentation Techniques

#### LLM-Based Paraphrasing
- **Multi-Model Rotation**: NVIDIA API (primary) + Gemini (fallback)
- **Difficulty Levels**: Easy vs. Hard paraphrasing modes
- **Medical Context Preservation**: Maintains clinical terminology accuracy
- **Configurable Ratios**: User-defined augmentation percentages (0.0-1.0)

#### Back-Translation Augmentation
- **Multi-Language Support**: German as intermediate language
- **Meaning Preservation**: Maintains semantic accuracy through translation cycles
- **Fallback Mechanisms**: Automatic retry with alternative models
- **Quality Control**: Length and content validation

#### Style Standardization
- **Clinical Voice Enforcement**: Neutral, professional medical tone
- **Absolute Language Removal**: Replaces guarantees with probabilistic language
- **Forum Sign-off Removal**: Eliminates informal communication patterns
- **Consistent Punctuation**: Standardized sentence structure

### 4. Data Quality Assurance

#### De-identification (PHI Removal)
- **Email Redaction**: `[REDACTED_EMAIL]` placeholder
- **Phone Number Masking**: `[REDACTED_PHONE]` placeholder
- **URL/IP Address Removal**: `[REDACTED_URL]` and `[REDACTED_IP]` placeholders
- **Configurable Privacy**: Optional PHI removal per dataset

#### Deduplication
- **Fingerprinting Algorithm**: MD5-based content hashing
- **Multi-Field Matching**: Instruction + Input + Output combination
- **Normalized Comparison**: Case-insensitive, whitespace-normalized matching
- **Performance Optimized**: In-memory set-based deduplication

#### Consistency Validation
- **LLM-Based QA Check**: Automated answer validation against context
- **Configurable Sampling**: Ratio-based consistency checking (e.g., 0.01)
- **Medical Safety Validation**: Ensures clinical accuracy and safety
- **Failure Tagging**: Marks samples with consistency issues

### 5. Advanced Augmentation Features

#### Knowledge Distillation
- **Pseudo-Label Generation**: Creates labels for unlabeled data
- **Fractional Processing**: Configurable percentage for distillation
- **Single-Prompt Approach**: Efficient single LLM call per sample
- **Length Control**: Maintains reasonable output lengths

#### Multi-Variant Generation
- **Configurable Counts**: 1-3 augmented variants per sample
- **Tagged Augmentations**: Tracks applied augmentation techniques
- **Original Preservation**: Always maintains base sample
- **Randomized IDs**: Unique identifiers for augmented variants

### 6. Output Generation & Storage

#### Centralized Format
- **SFT Schema**: Standardized Supervised Fine-Tuning format
- **Metadata Preservation**: Source, task type, and augmentation tags
- **Dual Output**: Simultaneous JSONL and CSV generation
- **Memory-Safe Streaming**: Handles large datasets efficiently

#### Storage Integration
- **Local Caching**: `cache/outputs/` directory storage
- **Google Drive Upload**: Automated cloud storage integration
- **Timestamped Naming**: Unique file identification
- **MIME Type Handling**: Proper content type specification

## ⚙️ Configuration Options

### Augmentation Parameters
```python
class AugmentOptions:
    paraphrase_ratio: float = 0.0          # 0.0-1.0
    paraphrase_outputs: bool = False       # Augment model answers
    backtranslate_ratio: float = 0.0       # 0.0-1.0
    style_standardize: bool = True         # Enforce clinical style
    deidentify: bool = True                # Remove PHI
    dedupe: bool = True                    # Remove duplicates
    max_chars: int = 5000                  # Text length limit
    consistency_check_ratio: float = 0.0   # 0.0-1.0
    distill_fraction: float = 0.0          # 0.0-1.0 for unlabeled
    expand: bool = True                    # Enable augmentation
    max_aug_per_sample: int = 2            # 1-3 variants
```

### Processing Parameters
```python
class ProcessParams:
    augment: AugmentOptions                # Augmentation settings
    sample_limit: Optional[int] = None     # Dataset sampling
    seed: int = 42                        # Reproducibility
```

## 📈 Performance & Monitoring

### Progress Tracking
- **Real-time Updates**: Live progress percentage and status messages
- **Background Processing**: Non-blocking job execution
- **State Management**: Thread-safe status tracking
- **Error Handling**: Comprehensive exception logging

### Resource Management
- **API Key Rotation**: Automatic fallback between multiple API keys
- **Rate Limiting**: Configurable request throttling
- **Memory Optimization**: Streaming processing for large datasets
- **Concurrent Processing**: Background task execution

## 🔒 Security & Privacy

### Data Protection
- **PHI Removal**: Automatic sensitive information redaction
- **Secure Storage**: Google Drive integration with OAuth2
- **Access Control**: Environment-based API key management
- **Audit Logging**: Comprehensive processing logs

### API Security
- **OAuth2 Integration**: Google Drive authentication
- **Token Management**: Secure credential handling
- **Request Validation**: Pydantic model validation
- **Error Sanitization**: Safe error message handling

## 🚀 Usage Examples

### Basic Processing
```bash
# Process HealthCareMagic with default settings
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"augment": {"paraphrase_ratio": 0.1}}' \
  https://binkhoale1812-medai-processing.hf.space/process/healthcaremagic
```

### Advanced Augmentation
```bash
# Process with comprehensive augmentation
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
      "max_chars": 5000,
      "consistency_check_ratio": 0.01,
      "max_aug_per_sample": 3
    },
    "sample_limit": 1000,
    "seed": 42
  }' \
  https://binkhoale1812-medai-processing.hf.space/process/icliniq
```

## 📊 Output Statistics

### Processing Metrics
- **Written Rows**: Total processed samples
- **Paraphrased Inputs**: Count of augmented user inputs
- **Paraphrased Outputs**: Count of augmented model responses
- **Back-translated**: Count of translation-augmented samples
- **Deduplication**: Count of skipped duplicate samples
- **Consistency Failures**: Count of validation failures

### File Outputs
- **JSONL Format**: Structured fine-tuning data with metadata
- **CSV Format**: Simplified tabular representation
- **Google Drive**: Cloud storage with automatic upload
- **Local Cache**: Persistent local storage

## 🔮 Future Enhancements

### Planned Features
- **Additional Dataset Support**: More medical dataset types
- **Advanced Augmentation**: More sophisticated LLM techniques
- **Quality Metrics**: Automated data quality scoring
- **Batch Processing**: Multiple dataset concurrent processing
- **Custom Schemas**: User-defined output formats

### Scalability Improvements
- **Distributed Processing**: Multi-node processing support
- **Streaming Augmentation**: Real-time data enhancement
- **Caching Optimization**: Improved performance and cost efficiency
- **API Rate Limiting**: Better resource management

## 📚 Technical Dependencies

### Core Libraries
- **FastAPI**: Web framework for API development
- **Hugging Face Hub**: Dataset downloading and management
- **Google GenAI**: Gemini model integration
- **ftfy**: Text encoding and normalization
- **langid**: Language detection
- **orjson**: High-performance JSON processing

### External Services
- **NVIDIA API**: Primary LLM service for paraphrasing
- **Google Gemini**: Fallback LLM service
- **Google Drive**: Cloud storage integration
- **Hugging Face Spaces**: Deployment platform

---

*This document provides a comprehensive overview of all data processing techniques implemented in the MedAI Processing project. For specific implementation details, refer to the individual module files in the `utils/` directory.*
