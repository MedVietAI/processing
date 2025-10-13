a. LLM-Based Paraphrasing 
- **Multi-model approach**: Llama-8B (same architecture) and Gemini (Flash/Pro) models for reliability 
- **Difficulty levels**: Easy vs. Hard paraphrasing modes to effectively use different models with auditing. 
- **Medical context preservation**: Maintains clinical terminology accuracy 
- **Configurable ratios**: User-defined augmentation percentages 

b. Back-Translation Augmentation 
- **Pivot languages** EN-VI-EN-VI...
- **Quality control**: Length and semantic similarity validation
- **Meaning preservation**: Maintains semantic accuracy through translation cycles 

c. Style Standardization 
- **Clinical voice enforcement**: Neutral, professional medical tone 
- **Absolute language removal**: Replaces guarantees with probabilistic language 
- **Forum sign-off removal**: Eliminates informal communication patterns 

d. Multi-Variant Generation (for reasoning) 
- **Answer variants**: Concise, detailed, clinical, patient-friendly styles 
- **Question variants**: Clarifying, follow-up, symptom-focused, treatment-focused 
- **Cross combinations**: All question × answer variant combinations (up to 9 per sample) e. Clinical Scenario Creation 
- **Context variations**: Emergency room, routine checkup, chronic conditions, family member perspectives
- **Enhanced diversity**: Multiple reasoning paths for improved model training 

f. Quality Assurance 
f1. Data Cleaning 
- **PHI removal**: Email, phone, URL, IP address redaction 
- **Deduplication**: MD5-based content hashing with normalized comparison 
- **Invalid response handling**: Detection and retry logic for failed responses 
- **Conversational element cleaning**: Removal of greetings and non-medical content 

f2. Validation 
- **Medical accuracy validation**: LLM-based consistency checking 
- **Length control**: Configurable maximum character limits 
- **Language detection**: English validation for content quality 

g. Output Formats: SFT Format 
- **Instruction**: Task description 
- **Input**: User question/context 
- **Output**: Model response
- **Metadata**: Augmentation tags and source information