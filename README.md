# GitHub Code Tutor 🚀

An intelligent Streamlit application that helps you understand and analyze GitHub repositories using AI. The app clones repositories, processes their code, and allows you to have interactive conversations about the codebase using Google's Gemini AI.

## Features ✨

- 🔍 Clone and analyze GitHub repositories
- 💬 Interactive chat interface with AI about the codebase
- 📊 Vector database storage for efficient code search
- 💾 Export chat history in multiple formats (JSON, CSV, Text)
- 🎨 Clean, user-friendly interface
- 🔄 Support for analyzing multiple repositories
- 📝 Persistent chat history with context

## Prerequisites 📋

- Python 3.12+
- Git installed on your system
- Google API key for Gemini AI

## Installation 🛠️

1. Clone this repository:
```bash
git clone https://github.com/yourusername/github-code-tutor.git
cd github-code-tutor
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Create a config.py file with your Google API key:
```python
import os

# Google Gemini API Key
GOOGLE_API_KEY = "your-google-api-key-here"  # Or use environment variable
# GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]

# Database configuration
LANCEDB_URI = os.path.join(os.getcwd(), "lancedb")

# Chunking settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
```

## Usage 🚀

1. Start the Streamlit app:
```bash
streamlit run codey.py
```

2. Enter a GitHub repository URL in the input field.

3. Click "Analyze" to process the repository.

4. Start asking questions about the codebase in the chat interface!

## Features in Detail 🔍

### Repository Analysis
- Clones GitHub repositories locally
- Processes and chunks code for efficient analysis
- Stores code embeddings in LanceDB for semantic search

### Chat Interface
- Interactive chat with Gemini AI about the codebase
- Context-aware responses based on code content
- Multi-turn conversation support
- Chat history persistence

### Export Options
- Export chat history in JSON format
- Export chat history in CSV format
- Export chat history as plain text

### State Management
- Clear chat history option
- Switch between different repositories
- Persistent storage of embeddings

## Project Structure 📁

```
.
├── codey.py                # Main Streamlit application
├── config.py              # Configuration settings
├── requirements.txt       # Project dependencies
└── services/
    ├── code_analyzer.py   # Code analysis and AI interaction
    └── streamlit_helpers.py # Streamlit UI helpers
```

## Configuration ⚙️

The app can be configured through the following settings in `config.py`:

- `GOOGLE_API_KEY`: Your Google API key for Gemini
- `LANCEDB_URI`: Location for the vector database storage
- `CHUNK_SIZE`: Size of code chunks for processing
- `CHUNK_OVERLAP`: Overlap between code chunks

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request.

## License 📄

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments 🙏

- [Streamlit](https://streamlit.io/) for the amazing web framework
- [Google Gemini](https://deepmind.google/technologies/gemini/) for the AI capabilities
- [LanceDB](https://lancedb.github.io/lancedb/) for vector storage

## Support 💬

If you encounter any issues or have questions, please file an issue on the GitHub repository.