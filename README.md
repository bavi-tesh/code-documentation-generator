# Code Documentation & Test Case Generator

An **AI-powered automation tool** that scans your source code and generates high-quality documentation and test cases — saving hours of manual work.

---

## Features

- Generates structured **code documentation** (Markdown or text)
- Auto-generates **test cases** for functions and classes
- Recursively scans entire directories for code files
- Supports flexible output formats and templates
- Easy to customize and integrate into existing projects

---

## Repository Structure

```
code-documentation-generator/
├── codes/              # Source code files to be documented
├── outputs/            # Generated documentation / test outputs
├── text/               # Template / prompt text files
├── requirements.txt    # Python dependencies (if applicable)
└── README.md           # Project documentation
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- (Optional) API key for LLM backend (e.g. OpenAI, Anthropic)
- Dependencies from `requirements.txt`

### Installation

```bash
git clone https://github.com/bavi-tesh/code-documentation-generator.git
cd code-documentation-generator
pip install -r requirements.txt
```

### (Optional) Set API Key

If you are using an AI model:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

---

## Usage

Run the generator on your codebase:

```bash
python generate_docs.py --source codes/ --output outputs/ --template text/template.md
```

**Arguments:**
- `--source` → Directory or file path containing source code  
- `--output` → Output folder for documentation and tests  
- `--template` → Optional custom template file  

After execution, you’ll find the generated docs and tests inside the `outputs/` directory.

---

## How It Works

1. **Parse & Extract** → Reads code structure (functions, classes, docstrings)  
2. **Analyze & Interpret** → Uses logic or AI to summarize behavior  
3. **Generate Docs** → Creates Markdown-formatted documentation  
4. **Generate Tests** → Builds example test cases for each function  
5. **Write Outputs** → Saves files in the `outputs/` folder  

---

## Customization

You can tailor the generated documentation to match your project’s style.

- Modify template files in the `text/` folder  
- Change formatting, section order, and header structure  
- Exclude or include specific directories or file types  
- Integrate into CI/CD pipelines for auto-generation  

---

## Example Output

### Function Documentation

```markdown
### def add(a: int, b: int) -> int

**Description:** Adds two numbers and returns the result.

**Parameters:**
- `a` (int): First number  
- `b` (int): Second number  

**Returns:**
- (int) Sum of the inputs  

**Example:**
```python
>>> add(3, 4)
7
```
```

### Generated Test File

```python
def test_add():
    assert add(2, 3) == 5
```

---

## Roadmap

- [ ] Multi-language support (JavaScript, C++, Java)  
- [ ] Diagram generation (flowcharts, dependency graphs)  
- [ ] Integration with MkDocs / Sphinx  
- [ ] GUI for easy documentation visualization  
- [ ] Continuous Documentation in CI/CD pipelines  

---

## Author

**Bavitesh M**  
GitHub: [@bavi-tesh](https://github.com/bavi-tesh)  
Reach out for feedback, ideas, or collaborations!

