# C3PO - INFO to Text Converter

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.6%2B-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Dependency Management: uv](https://img.shields.io/badge/dependency%20management-uv-blue.svg)](https://github.com/astral-sh/uv)

A powerful and efficient tool for converting INFO format files to clean text format while preserving content integrity.

## Features

- **Efficient Processing**: Handles large files with parallel processing
- **Content Validation**: Ensures the converted output maintains content integrity
- **Format Cleaning**: Removes headers, separators, menu items, and control characters
- **Comprehensive Logging**: Detailed logging system with configurable levels
- **Customizable**: Adjustable parameters for processing and validation

## Installation

### Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/c3po.git
cd c3po

# Create and activate virtual environment with uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/yourusername/c3po.git
cd c3po

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python c3po.py input_file.info output_file.txt [options]
```

### Options

- `--no-verify`: Skip content validation
- `--workers=N`: Set the number of worker threads (default: CPU count)
- `--debug`: Enable debug mode
- `--log=LEVEL`: Set log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

### Examples

Basic conversion:
```bash
python c3po.py document.info document.txt
```

Convert with 4 worker threads and skip validation:
```bash
python c3po.py large_document.info large_document.txt --workers=4 --no-verify
```

Convert with debug mode and detailed logging:
```bash
python c3po.py document.info document.txt --debug --log=DEBUG
```

## Content Validation

The tool validates converted content using several metrics:

- **Paragraph Ratio**: Ensures paragraph structure is maintained (ideal: 0.5-1.5)
- **Content Length Ratio**: Checks overall content length preservation (ideal: 0.7-1.2)
- **Keyword Retention**: Verifies important keywords are preserved (ideal: >25%)
- **Content Similarity**: Samples content to ensure similarity (ideal: >10%)

## Performance

- Processes files at approximately 1.5MB/second on modern hardware
- Memory usage scales efficiently with file size through chunked processing
- Parallel processing utilizes available CPU cores for optimal performance

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The tqdm library for progress bar functionality
- Python's concurrent.futures for parallel processing capabilities
