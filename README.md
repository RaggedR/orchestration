# Orchestration

Experiments testing how LLM multi-agent migration topology affects solution diversity (none/ring/star/fully-connected), extending the "From Games to Graphs" paper. Key finding: the ordering inverts vs genetic algorithms, explained via contravariant functoriality (left vs right Kan extensions).

## Tech Stack
- Python 3 (OpenAI API, sentence-transformers)
- Haskell (formal theory)

## Setup
```bash
pip install -r requirements.txt
python run_experiment.py
```

## License
MIT
