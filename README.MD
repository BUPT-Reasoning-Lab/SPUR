# Readme

# Decoding Scientific Experimental Images:  The SPUR Benchmark for Perception, Understanding, and Reasoning

## Directory Structure

```
SPUR/
├── api_test/               # Stores core code for LLM API testing, including various test scripts, a base async testing framework, and test submodules for different models/strategies
│   ├── base.py	     # Main experiment script
│   ├── Cantor.py        # MCOT experiment script for Cantor
│   ├── VoT.py            # MCOT experiment script for VoT
│   ├── DDCoT.py       # MCOT experiment script for DDCoT
│   └── VIC/
│       ├── VIC_test.py  # MCOT experiment script for VIC
├── dataset/                # Stores raw datasets required for multimodal QA tasks, including JSON files of single/multi-panel QA pairs, reasoning-based QA pair files, and supporting original image resources
│   ├── origin_images/
│   ├── single-panel_qa_pair.json
│   ├── multi-panel_trend_qa_pair.json
│   └── reason_qa_pair.json
├── sub_panel/              # Exclusive module for processing multi-panel QA tasks, containing utility code for multi-panel dataset processing, panel splitting and validation
│   └── cut_picture.py   # panel detect script
└── data_prepar/            # Data preprocessing module, covering QA pair generation, prompt template definition for different types of questions, and code for dataset cleaning/standardization
    ├── qa_pair_generation.py
    ├── prompt_MDP.txt
    ├── prompt_reasoning.txt
    └── prompt_CPU.txt

```

## Key Components

### 1. QA Pair Generation

The `qa_pair_generation.py` script generates question-answer pairs based on scientific images using GPT-4o. It supports three modes:

- `single`: Generates single-panelquestions (numerical, graphical, spatial)
- `multi_trend`: Generates cross-panel trend understanding questions
- `reason`: Generates reasoning-based questions (qualitative and quantitative)

### Usage

```python
# In qa_pair_generation.py, set the mode and run
Mode = "single"  # or "multi_trend", "reason"

```

### 2. Model Testing

The `api_test` directory contains scripts for testing models on the generated QA pairs:

- `base.py`: Provides the core asynchronous testing framework
- `doubao.py`: Implements testing specifically for the Doubao model

### Usage

```bash
# Run the testing script with a specific mode
python api_test/doubao.py
# (Modify the mode in the main function: "single", "multi_trend", or "reason")

```

### 3. Question Types

SPUR includes various question types to assess different aspects of scientific image understanding:

### Single-panel Perception Questions

- [Single-image - Numerical]: Focus on data values, significance, and coordinate logic in charts
- [Single-image - Morphological]: Target cell morphology and staining patterns in microscopy images
- [Single-image - Spatial]: Evaluate understanding of band arrangements in WB/electrophoresis images

### Cross-Panel Understanding Questions

- [Trend - Numerical]: Assess cross-panel understanding of quantitative data
- [Trend  - Morphological]: Test consistency understanding across staining/pathology images
- [Trend  - Spatial]: Evaluate spatial relationship comprehension across WB panels
- [Cross-Modality relation]: Test understanding of relationships between different image types

### Reasoning Questions

- [Reason_Quantitative]: Require quantitative analysis across multiple images
- [Reason_Qualitative]: Focus on qualitative reasoning about experimental processes

## Output

The testing scripts generate:

- JSON files with detailed results in `results/` directory
- CSV files with summarized results for easy analysis
- Log files documenting the testing process

## Requirements

- Python 3.8+
- OpenAI Python library
- pandas
- aiolimiter
- tqdm

## Notes

- The benchmark requires access to scientific images and their corresponding annotations
- API keys for GPT-4o and Doubao models need to be configured in the respective scripts
- The number of concurrent requests (RPM) can be adjusted in the Config class

This benchmark is designed to expose model vulnerabilities in scientific image interpretation, including issues with cross-panel comparison, scale interpretation, and spatial relationship understanding.

### Updated Additions (if applicable, adjust based on actual changes; if no specific change details provided, the above remains consistent as per the request to keep unchanged parts intact)

*No additional changes specified—this README retains all original content while adhering to formatting and language requirements.*
